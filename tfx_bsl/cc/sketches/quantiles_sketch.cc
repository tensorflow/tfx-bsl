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
  QuantilesSketchImpl(
      double eps, int64_t max_num_elements,
      tensorflow::boosted_trees::quantiles::WeightedQuantilesSummary<double,
                                                                     double>
          summary = {})
      : eps_(eps),
        max_num_elements_(max_num_elements),
        stream_(absl::nullopt),
        staged_summary_(std::move(summary)) {}

  QuantilesSketchImpl(const QuantilesSketchImpl&) = delete;
  QuantilesSketchImpl& operator=(const QuantilesSketchImpl&) = delete;

  Status AddWeightedValues(const arrow::DoubleArray& values,
                           const arrow::DoubleArray& weights) {
    CreateStream();
    for (int i = 0; i < values.length(); ++i) {
      if (values.IsNull(i) || weights.IsNull(i) || weights.Value(i) <= 0) {
        continue;
      }
      stream_->PushEntry(values.Value(i), weights.Value(i));
    }
    return Status::OK();
  }

  Status AddValues(const arrow::DoubleArray& values) {
    CreateStream();
    for (int i = 0; i < values.length(); ++i) {
      if (values.IsNull(i)) continue;
      stream_->PushEntry(values.Value(i), 1.0);
    }
    return Status::OK();
  }

  // TODO(zhuo, iindyk): determine whether preserve all summary levels by
  // calling stream_->SerializeInternalSummaries().
  std::string Serialize() {
    Quantiles sketch_proto;
    sketch_proto.set_eps(eps_);
    sketch_proto.set_max_num_elements(max_num_elements_);
    const Summary& summary = GetSummary();
    for (const auto& s : summary.GetEntryList()) {
      sketch_proto.add_value(s.value);
      sketch_proto.add_weight(s.weight);
      sketch_proto.add_min_rank(s.min_rank);
      sketch_proto.add_max_rank(s.max_rank);
    }
    return sketch_proto.SerializeAsString();
  }

  static std::unique_ptr<QuantilesSketchImpl> Deserialize(
      absl::string_view serialized) {
    Quantiles sketch_proto;
    sketch_proto.ParseFromArray(serialized.data(), serialized.size());
    size_t num_summaries = sketch_proto.value_size();
    std::vector<SummaryEntry> summary_entries;
    summary_entries.reserve(num_summaries);
    for (int i = 0; i < num_summaries; ++i) {
      summary_entries.push_back(
          SummaryEntry(sketch_proto.value(i), sketch_proto.weight(i),
                       sketch_proto.min_rank(i), sketch_proto.max_rank(i)));
    }
    Summary summary;
    summary.BuildFromSummaryEntries(summary_entries);
    return absl::make_unique<QuantilesSketchImpl>(
        sketch_proto.eps(), sketch_proto.max_num_elements(),
        std::move(summary));
  }

  void Merge(QuantilesSketchImpl& other) {
    CreateStream();
    const Summary& other_summary = other.GetSummary();
    stream_->PushSummary(other_summary.GetEntryList());
  }

  Summary GetSummary() {
    // The stream state is destoryed after Finalize(). But a stream can be
    // recreated from the staged_summary_.
    if (stream_) {
      stream_->Finalize();
      staged_summary_ = stream_->GetFinalSummary();
      stream_ = absl::nullopt;
    }
    return staged_summary_;
  }

 private:
  void CreateStream() {
    if (!stream_) {
      stream_ = Stream(eps_, max_num_elements_);
      if (staged_summary_.Size() > 0) {
        stream_->PushSummary(staged_summary_.GetEntryList());
        staged_summary_.Clear();
      }
    }
  }

  double eps_;
  int64_t max_num_elements_;

  // stream_ may be nullopt due to the fact that stream_->Finalize() can only
  // be called once. So once a stream_ is finalized, it will be dumped into
  // `staged_summary_`. Later if a stream is needed, a new one will be created
  // from `staged_summary_`.
  absl::optional<Stream> stream_;
  Summary staged_summary_;
};

QuantilesSketch::QuantilesSketch(double eps, int64_t max_num_elements)
    : impl_(absl::make_unique<QuantilesSketchImpl>(eps, max_num_elements)) {}

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
  return Status::OK();
}

void QuantilesSketch::Merge(QuantilesSketch& other) {
  impl_->Merge(*other.impl_);
}

Status QuantilesSketch::GetQuantiles(int64_t num_quantiles,
                                     std::shared_ptr<arrow::Array>* quantiles) {
  Summary summary = impl_->GetSummary();
  std::vector<double> quantiles_vec = summary.GenerateQuantiles(num_quantiles);
  arrow::DoubleBuilder result_builder;
  TFX_BSL_RETURN_IF_ERROR(
      FromArrowStatus(result_builder.AppendValues(quantiles_vec)));
  TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(result_builder.Finish(quantiles)));
  return Status::OK();
}

std::string QuantilesSketch::Serialize() {
  return impl_->Serialize();
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
