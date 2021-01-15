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

}  // namespace

class QuantilesSketchImpl {
 public:
  QuantilesSketchImpl(double eps, int64_t max_num_elements, int64_t num_streams,
                      bool compacted = false,
                      std::vector<std::vector<Summary>> summaries = {},
                      std::vector<std::vector<BufferEntry>> buffer_entries = {})
      : eps_(eps),
        max_num_elements_(max_num_elements),
        num_streams_(num_streams),
        compacted_(compacted) {
    // Create streams
    streams_.reserve(num_streams_);
    for (int i = 0; i < num_streams_; ++i) {
      streams_.push_back(Stream(eps_, max_num_elements_));
    }

    // Recover local summaries.
    if (!summaries.empty()) {
      for (int i = 0; i < num_streams_; ++i) {
        if (!summaries[i].empty()) {
          streams_[i].SetInternalSummaries(summaries[i]);
        }
      }
    }

    // Recover buffer elements.
    if (!buffer_entries.empty()) {
      for (int i = 0; i < num_streams_; ++i) {
        for (auto& entry : buffer_entries[i]) {
          streams_[i].PushEntry(entry.value, entry.weight);
        }
      }
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

    const int num_inputs_per_stream = weights.length();
    for (int stream_idx = 0; stream_idx < num_streams_; ++stream_idx) {
      Stream& stream = streams_[stream_idx];
      for (int value_idx = stream_idx, weight_idx = 0;
           weight_idx < num_inputs_per_stream;
           value_idx += num_streams_, ++weight_idx) {
        if (values.IsNull(value_idx) || weights.IsNull(weight_idx) ||
            weights.Value(weight_idx) <= 0) {
          continue;
        }
        stream.PushEntry(values.Value(value_idx), weights.Value(weight_idx));
      }
    }
    return Status::OK();
  }

  Status AddValues(const arrow::DoubleArray& values) {
    if (finalized_) {
      return errors::FailedPrecondition(
          "Attempting to add values to a finalized sketch.");
    }

    for (int stream_idx = 0; stream_idx < num_streams_; ++stream_idx) {
      Stream& stream = streams_[stream_idx];
      for (int value_idx = stream_idx; value_idx < values.length();
           value_idx += num_streams_) {
        if (values.IsNull(value_idx)) continue;
        stream.PushEntry(values.Value(value_idx), 1.0);
      }
    }
    return Status::OK();
  }

  Status Compact() {
    if (finalized_) {
      return errors::FailedPrecondition(
          "Attempting to compact a finalized sketch.");
    }
    if (!compacted_) {
      for (auto& stream : streams_) {
        stream.Finalize();
        std::vector<SummaryEntry> final_summary_entries =
            stream.GetFinalSummary().GetEntryList();
        stream = Stream(eps_, max_num_elements_);
        stream.PushSummary(std::move(final_summary_entries));
      }
      compacted_ = true;
    }
    return Status::OK();
  }

  Status Serialize(std::string& serialized) const {
    if (finalized_) {
      return errors::FailedPrecondition(
          "Attempting to serialize a finalized sketch.");
    }

    Quantiles sketch_proto;
    for (auto& stream : streams_) {
      Quantiles::Stream* stream_proto = sketch_proto.add_streams();
      stream_proto->set_eps(eps_);
      stream_proto->set_max_num_elements(max_num_elements_);
      stream_proto->set_compacted(compacted_);

      // Add local summaries.
      const std::vector<Summary>& summaries = stream.GetInternalSummaries();
      for (const auto& summary : summaries) {
        Quantiles::Stream::Summary* summary_proto =
            stream_proto->add_summaries();
        for (const auto& entry : summary.GetEntryList()) {
          summary_proto->add_value(entry.value);
          summary_proto->add_weight(entry.weight);
          summary_proto->add_min_rank(entry.min_rank);
          summary_proto->add_max_rank(entry.max_rank);
        }
      }

      // Add buffer elements.
      const std::vector<BufferEntry>& buffer_entries =
          stream.GetBufferEntryList();
      Quantiles::Stream::Buffer* buffer_proto = stream_proto->mutable_buffer();
      for (const auto& entry : buffer_entries) {
        buffer_proto->add_value(entry.value);
        buffer_proto->add_weight(entry.weight);
      }
    }

    serialized = sketch_proto.SerializeAsString();
    return Status::OK();
  }

  static Status Deserialize(absl::string_view serialized,
                            std::unique_ptr<QuantilesSketchImpl>* result) {
    Quantiles sketch_proto;
    sketch_proto.ParseFromArray(serialized.data(), serialized.size());
    std::vector<std::vector<Summary>> summaries;
    std::vector<std::vector<BufferEntry>> buffer_entries;
    const size_t num_streams = sketch_proto.streams_size();
    if (num_streams < 1) {
      return errors::InvalidArgument("Serialized sketch has no streams.");
    }

    summaries.reserve(num_streams);
    buffer_entries.reserve(num_streams);
    const double eps = sketch_proto.streams(0).eps();
    const int64_t max_num_elements = sketch_proto.streams(0).max_num_elements();
    const bool compacted = sketch_proto.streams(0).compacted();

    for (int stream_idx = 0; stream_idx < num_streams; ++stream_idx) {
      const Quantiles::Stream& stream_proto = sketch_proto.streams(stream_idx);

      // Recover summaries.
      std::vector<Summary> stream_summaries;
      const size_t num_summaries = stream_proto.summaries_size();
      stream_summaries.reserve(num_summaries);
      for (int i = 0; i < num_summaries; ++i) {
        const Quantiles::Stream::Summary& summary_proto =
            stream_proto.summaries(i);
        std::vector<SummaryEntry> summary_entries;
        const size_t num_summary_entries = summary_proto.value_size();
        summary_entries.reserve(num_summary_entries);
        for (int j = 0; j < num_summary_entries; ++j) {
          summary_entries.push_back(SummaryEntry(
              summary_proto.value(j), summary_proto.weight(j),
              summary_proto.min_rank(j), summary_proto.max_rank(j)));
        }
        Summary summary;
        summary.BuildFromSummaryEntries(summary_entries);
        stream_summaries.push_back(summary);
      }
      summaries.push_back(std::move(stream_summaries));

      // Recover buffer.
      const Quantiles::Stream::Buffer& buffer_proto = stream_proto.buffer();
      std::vector<BufferEntry> stream_buffer_entries;
      const size_t num_buffer_entries = buffer_proto.value_size();
      stream_buffer_entries.reserve(num_buffer_entries);
      for (int i = 0; i < num_buffer_entries; ++i) {
        stream_buffer_entries.push_back(
            BufferEntry(buffer_proto.value(i), buffer_proto.weight(i)));
      }
      buffer_entries.push_back(std::move(stream_buffer_entries));
    }

    *result = absl::make_unique<QuantilesSketchImpl>(
        eps, max_num_elements, num_streams, compacted, std::move(summaries),
        std::move(buffer_entries));
    return Status::OK();
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
    if (num_streams_ != other.num_streams_) {
      return errors::FailedPrecondition(
          "Attempting to merge sketches with different number of streams.");
    }
    for (int i = 0; i < num_streams_; i++) {
      streams_[i].PushStream(other.streams_[i]);
    }
    if (other.compacted_) {
      compacted_ = true;
    }
    return Status::OK();
  }

  std::vector<Summary> GetFinalSummaries() {
    // The stream state is destroyed after Finalize().
    if (!finalized_) {
      for (auto& stream : streams_) {
        stream.Finalize();
      }
      finalized_ = true;
    }
    std::vector<Summary> final_summaries;
    final_summaries.reserve(num_streams_);
    for (auto& stream : streams_) {
      final_summaries.push_back(stream.GetFinalSummary());
    }
    return final_summaries;
  }

  int64_t num_streams() const { return num_streams_; }

 private:
  const double eps_;
  const int64_t max_num_elements_;
  const int64_t num_streams_;
  std::vector<Stream> streams_;
  bool finalized_ = false;
  bool compacted_ = false;
};

// static
Status QuantilesSketch::Make(double eps, int64_t max_num_elements,
                             int64_t num_streams,
                             std::unique_ptr<QuantilesSketch>* result) {
  if (eps <= 0) {
    return errors::InvalidArgument("eps must be positive.");
  }
  if (max_num_elements < 1) {
    return errors::InvalidArgument("max_num_elements must be >= 1.");
  }
  if (num_streams < 1) {
    return errors::InvalidArgument("num_streams must be >= 1.");
  }
  // Error bound is adjusted by height of the computation graph. Note that the
  // current implementation always has height of 2: one level from `AddValues`,
  // `AddWeightedValues` and `Merge` that perform multi-level summary
  // compression maintaining the error bound, and another level from
  // `GetQuantiles` that performs final summary compression adding to the final
  // error bound. Final summary compression can only be triggered once. See
  // weighted_quantiles_stream.h for details.
  *result = absl::WrapUnique(
      new QuantilesSketch(absl::make_unique<QuantilesSketchImpl>(
          eps / 2, max_num_elements, num_streams)));
  return Status::OK();
}

QuantilesSketch::~QuantilesSketch() {}
QuantilesSketch::QuantilesSketch(QuantilesSketch&&) = default;
QuantilesSketch& QuantilesSketch::operator=(QuantilesSketch&&) = default;

Status QuantilesSketch::AddWeightedValues(
    std::shared_ptr<arrow::Array> values,
    std::shared_ptr<arrow::Array> weights) {
  if (values->length() != weights->length() * impl_->num_streams()) {
    return errors::InvalidArgument(
        "Values size must be equal to weights size times number of streams.");
  }
  TFX_BSL_RETURN_IF_ERROR(MaybeCastToDoubleArray(&values));
  TFX_BSL_RETURN_IF_ERROR(MaybeCastToDoubleArray(&weights));
  return impl_->AddWeightedValues(
      static_cast<const arrow::DoubleArray&>(*values),
      static_cast<const arrow::DoubleArray&>(*weights));
}

Status QuantilesSketch::AddValues(std::shared_ptr<arrow::Array> values) {
  if (values->length() % impl_->num_streams() != 0) {
    return errors::InvalidArgument(
        "Values size must be divisible by the number of streams.");
  }
  TFX_BSL_RETURN_IF_ERROR(MaybeCastToDoubleArray(&values));
  return impl_->AddValues(static_cast<const arrow::DoubleArray&>(*values));
}

Status QuantilesSketch::Compact() { return impl_->Compact(); }

Status QuantilesSketch::Merge(const QuantilesSketch& other) {
  return impl_->Merge(*other.impl_);
}

Status QuantilesSketch::GetQuantiles(int64_t num_quantiles,
                                     std::shared_ptr<arrow::Array>* quantiles) {
  if (num_quantiles <= 1) {
    return errors::InvalidArgument(
        "Number of requested quantiles must be >= 2.");
  }
  // Extract final summaries and generate quantiles.
  std::vector<Summary> final_summaries = impl_->GetFinalSummaries();
  std::vector<double> quantiles_vec;
  quantiles_vec.reserve(num_quantiles * impl_->num_streams());
  for (auto& summary : final_summaries) {
    auto summary_quantiles = summary.GenerateQuantiles(num_quantiles);
    quantiles_vec.insert(quantiles_vec.end(), summary_quantiles.begin(),
                         summary_quantiles.end());
  }

  // Convert to a `FixedSizeListArray` with result for each stream.
  arrow::DoubleBuilder result_builder;
  TFX_BSL_RETURN_IF_ERROR(
      FromArrowStatus(result_builder.AppendValues(quantiles_vec)));
  std::shared_ptr<arrow::Array> quantiles_flat;
  TFX_BSL_RETURN_IF_ERROR(
      FromArrowStatus(result_builder.Finish(&quantiles_flat)));
  TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(
      arrow::FixedSizeListArray::FromArrays(quantiles_flat, num_quantiles + 1)
          .Value(quantiles)));
  return Status::OK();
}

Status QuantilesSketch::Serialize(std::string& serialized) const {
  return impl_->Serialize(serialized);
}

// static
Status QuantilesSketch::Deserialize(absl::string_view serialized,
                                    std::unique_ptr<QuantilesSketch>* result) {
  std::unique_ptr<QuantilesSketchImpl> impl;
  TFX_BSL_RETURN_IF_ERROR(QuantilesSketchImpl::Deserialize(serialized, &impl));
  *result = absl::WrapUnique(new QuantilesSketch(std::move(impl)));
  return Status::OK();
}

QuantilesSketch::QuantilesSketch(std::unique_ptr<QuantilesSketchImpl> impl)
    : impl_(std::move(impl)) {}

}  // namespace sketches
}  // namespace tfx_bsl
