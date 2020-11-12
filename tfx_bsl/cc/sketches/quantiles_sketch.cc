// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "tfx_bsl/cc/sketches/quantiles_sketch.h"

#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "arrow/compute/api.h"
#include "tfx_bsl/cc/sketches/weighted_quantiles_stream.h"
#include "tfx_bsl/cc/sketches/weighted_quantiles_summary.h"
#include "tfx_bsl/cc/util/status.h"
#include "tfx_bsl/cc/util/status_util.h"

namespace tfx_bsl {
namespace sketches {
namespace {
using Buffer = tensorflow::boosted_trees::quantiles::WeightedQuantilesBuffer<
    double, double, std::less<double>>;
using BufferEntry = Buffer::BufferEntry;
using Stream =
    tensorflow::boosted_trees::quantiles::WeightedQuantilesStream<double,
                                                                  double>;
using Summary = Stream::Summary;
using SummaryEntry = Stream::SummaryEntry;

// TODO(b/171748040): clean this up.
#if ARROW_VERSION_MAJOR < 1

Status MaybeCastToDoubleArray(std::shared_ptr<arrow::Array>* array) {
  if ((*array)->type()->id() == arrow::Type::DOUBLE) return Status::OK();

  arrow::compute::FunctionContext ctx(arrow::default_memory_pool());
  std::shared_ptr<arrow::Array> result;
  TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(arrow::compute::Cast(
      &ctx, **array, arrow::float64(),
      // Allow integer truncation (int64->float64).
      arrow::compute::CastOptions(/*safe=*/false), &result)));
  *array = std::move(result);
  return Status::OK();
}

#else

Status MaybeCastToDoubleArray(std::shared_ptr<arrow::Array>* array) {
  if ((*array)->type()->id() == arrow::Type::DOUBLE) return Status::OK();
  std::shared_ptr<arrow::Array> result;
  TFX_BSL_ASSIGN_OR_RETURN_ARROW(
      result, arrow::compute::Cast(**array, arrow::float64(),
                                   // Allow integer truncation (int64->float64).
                                   arrow::compute::CastOptions::Unsafe()));
  *array = std::move(result);
  return Status::OK();
}

#endif

}  // namespace

class QuantilesSketchImpl {
 public:
  QuantilesSketchImpl(double eps, int64_t max_num_elements,
                      std::vector<Summary> summaries = {},
                      std::vector<BufferEntry> buffer_entries = {})
      : eps_(eps),
        max_num_elements_(max_num_elements),
        stream_(Stream(eps_, max_num_elements_)) {
    // Recover local summaries.
    if (!summaries.empty()) {
      stream_.SetInternalSummaries(summaries);
    }

    // Recover buffer elements.
    for (auto& entry : buffer_entries) {
      stream_.PushEntry(entry.value, entry.weight);
    }
  }

  QuantilesSketchImpl(const QuantilesSketchImpl&) = delete;
  QuantilesSketchImpl& operator=(const QuantilesSketchImpl&) = delete;

  Status AddWeightedValues(const arrow::DoubleArray& values,
                           const arrow::DoubleArray& weights) {
    if (finalized_) {
      return errors::FailedPrecondition(
          "Attempting to add values to a finalized sketch.");
    }

    for (int i = 0; i < values.length(); ++i) {
      if (values.IsNull(i) || weights.IsNull(i) || weights.Value(i) <= 0) {
        continue;
      }
      stream_.PushEntry(values.Value(i), weights.Value(i));
    }
    return Status::OK();
  }

  Status AddValues(const arrow::DoubleArray& values) {
    if (finalized_) {
      return errors::FailedPrecondition(
          "Attempting to add values to a finalized sketch.");
    }

    for (int i = 0; i < values.length(); ++i) {
      if (values.IsNull(i)) continue;
      stream_.PushEntry(values.Value(i), 1.0);
    }
    return Status::OK();
  }

  Status Serialize(std::string& serialized) const {
    if (finalized_) {
      return errors::FailedPrecondition(
          "Attempting to serialize a finalized sketch.");
    }

    Quantiles sketch_proto;
    Quantiles::Stream* stream_proto = sketch_proto.add_streams();
    stream_proto->set_eps(eps_);
    stream_proto->set_max_num_elements(max_num_elements_);

    // Add local summaries.
    const std::vector<Summary>& summaries = stream_.GetInternalSummaries();
    for (const auto& summary : summaries) {
      Quantiles::Stream::Summary* summary_proto = stream_proto->add_summaries();
      for (const auto& entry : summary.GetEntryList()) {
        summary_proto->add_value(entry.value);
        summary_proto->add_weight(entry.weight);
        summary_proto->add_min_rank(entry.min_rank);
        summary_proto->add_max_rank(entry.max_rank);
      }
    }

    // Add buffer elements.
    const std::vector<BufferEntry>& buffer_entries =
        stream_.GetBufferEntryList();
    Quantiles::Stream::Buffer* buffer_proto = stream_proto->mutable_buffer();
    for (const auto& entry : buffer_entries) {
      buffer_proto->add_value(entry.value);
      buffer_proto->add_weight(entry.weight);
    }
    serialized = sketch_proto.SerializeAsString();
    return Status::OK();
  }

  static std::unique_ptr<QuantilesSketchImpl> Deserialize(
      absl::string_view serialized) {
    Quantiles sketch_proto;
    sketch_proto.ParseFromArray(serialized.data(), serialized.size());
    Quantiles::Stream stream_proto = sketch_proto.streams(0);

    // Recover summaries.
    std::vector<Summary> summaries;
    size_t num_summaries = stream_proto.summaries_size();
    summaries.reserve(num_summaries);
    for (int i = 0; i < num_summaries; ++i) {
      Quantiles::Stream::Summary summary_proto = stream_proto.summaries(i);
      std::vector<SummaryEntry> summary_entries;
      size_t num_summary_entries = summary_proto.value_size();
      summary_entries.reserve(num_summary_entries);
      for (int j = 0; j < num_summary_entries; ++j) {
        summary_entries.push_back(
            SummaryEntry(summary_proto.value(j), summary_proto.weight(j),
                         summary_proto.min_rank(j), summary_proto.max_rank(j)));
      }
      Summary summary;
      summary.BuildFromSummaryEntries(summary_entries);
      summaries.push_back(summary);
    }

    // Recover buffer.
    Quantiles::Stream::Buffer buffer_proto = stream_proto.buffer();
    std::vector<BufferEntry> buffer_entries;
    size_t num_buffer_entries = buffer_proto.value_size();
    buffer_entries.reserve(num_buffer_entries);
    for (int i = 0; i < num_buffer_entries; ++i) {
      buffer_entries.push_back(
          BufferEntry(buffer_proto.value(i), buffer_proto.weight(i)));
    }

    return absl::make_unique<QuantilesSketchImpl>(
        stream_proto.eps(), stream_proto.max_num_elements(),
        std::move(summaries), std::move(buffer_entries));
  }

  Status Merge(const QuantilesSketchImpl& other) {
    if (finalized_) {
      return errors::FailedPrecondition(
          "Attempting to merge to a finalized sketch.");
    }
    if (other.finalized_) {
      return errors::FailedPrecondition(
          "Attempting to merge a finalized sketch.");
    }

    stream_.PushStream(other.stream_);
    return Status::OK();
  }

  Summary GetFinalSummary() {
    // The stream state is destroyed after Finalize().
    if (!finalized_) {
      stream_.Finalize();
      finalized_ = true;
    }
    return stream_.GetFinalSummary();
  }

 private:
  double eps_;
  int64_t max_num_elements_;
  Stream stream_;
  bool finalized_ = false;
};

// Error bound is adjusted by height of the computation graph. Note that the
// current implementation always has height of 2: one level from `AddValues`,
// `AddWeightedValues` and `Merge` that perform multi-level summary compression
// maintaining the error bound, and another level from `GetQuantiles` that
// performs final summary compression adding to the final error bound. Final
// summary compression can only be triggered once. See
// weighted_quantiles_stream.h for details.
QuantilesSketch::QuantilesSketch(double eps, int64_t max_num_elements)
    : impl_(absl::make_unique<QuantilesSketchImpl>(eps / 2, max_num_elements)) {
}

QuantilesSketch::~QuantilesSketch() {}
QuantilesSketch::QuantilesSketch(QuantilesSketch&&) = default;
QuantilesSketch& QuantilesSketch::operator=(QuantilesSketch&&) = default;

Status QuantilesSketch::AddWeightedValues(
    std::shared_ptr<arrow::Array> values,
    std::shared_ptr<arrow::Array> weights) {
  if (values->length() != weights->length()) {
    return errors::InvalidArgument(
        "Values and weights arrays must be of the same size.");
  }
  TFX_BSL_RETURN_IF_ERROR(MaybeCastToDoubleArray(&values));
  TFX_BSL_RETURN_IF_ERROR(MaybeCastToDoubleArray(&weights));
  return impl_->AddWeightedValues(
      static_cast<const arrow::DoubleArray&>(*values),
      static_cast<const arrow::DoubleArray&>(*weights));
}

Status QuantilesSketch::AddValues(std::shared_ptr<arrow::Array> values) {
  TFX_BSL_RETURN_IF_ERROR(MaybeCastToDoubleArray(&values));
  return impl_->AddValues(static_cast<const arrow::DoubleArray&>(*values));
}

Status QuantilesSketch::Merge(const QuantilesSketch& other) {
  return impl_->Merge(*other.impl_);
}

Status QuantilesSketch::GetQuantiles(int64_t num_quantiles,
                                     std::shared_ptr<arrow::Array>* quantiles) {
  Summary summary = impl_->GetFinalSummary();
  std::vector<double> quantiles_vec = summary.GenerateQuantiles(num_quantiles);
  arrow::DoubleBuilder result_builder;
  TFX_BSL_RETURN_IF_ERROR(
      FromArrowStatus(result_builder.AppendValues(quantiles_vec)));
  TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(result_builder.Finish(quantiles)));
  return Status::OK();
}

Status QuantilesSketch::Serialize(std::string& serialized) const {
  return impl_->Serialize(serialized);
}

// static
QuantilesSketch QuantilesSketch::Deserialize(absl::string_view serialized) {
  auto impl = QuantilesSketchImpl::Deserialize(serialized);
  return QuantilesSketch(std::move(impl));
}

QuantilesSketch::QuantilesSketch(std::unique_ptr<QuantilesSketchImpl> impl)
    : impl_(std::move(impl)) {}

}  // namespace sketches
}  // namespace tfx_bsl
