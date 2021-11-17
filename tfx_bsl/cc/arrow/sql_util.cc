// Copyright 2021 Google LLC
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
#include "tfx_bsl/cc/arrow/sql_util.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "zetasql/public/catalog.h"
#include "zetasql/public/evaluator.h"
#include "zetasql/public/simple_catalog.h"
#include "zetasql/public/type.h"
#include "zetasql/public/type.pb.h"
#include "zetasql/public/value.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "arrow/api.h"
#include "arrow/array.h"
#include "tfx_bsl/cc/util/status_util.h"

namespace tfx_bsl {

namespace {
constexpr char kTableName[] = "Examples";

using zetasql::SimpleCatalog;
using zetasql::SimpleTable;

// Convert the sql query result to the slices list.
void ConvertQueryResultToSlices(
    std::unique_ptr<zetasql::EvaluatorTableIterator> query_result_iterator,
    std::vector<std::vector<std::vector<std::pair<std::string, std::string>>>>*
        result) {
  while (query_result_iterator->NextRow()) {
    // Only one column will be returned. So we use index 0 here.
    const zetasql::Value& value = query_result_iterator->GetValue(0);
    std::vector<std::vector<std::pair<std::string, std::string>>> row;
    row.reserve(value.num_elements());

    // The data format of the value is like:
    // ARRAY([
    //  STRUCT('slice_key_1': 1.1, 'slice_key_2': "male"),
    //  STRUCT('slice_key_1': 3.4, 'slice_key_2': "female"),
    //  STRUCT(...)
    // ])
    for (const zetasql::Value& struct_element : value.elements()) {
      std::vector<std::pair<std::string, std::string>> slices;
      for (int i = 0; i < struct_element.num_fields(); i++) {
        const zetasql::Value& field = struct_element.field(i);
        std::string slice_key =
            struct_element.type()->AsStruct()->field(i).name;
        std::string slice_value;
        // For string or byte type, field.ShortDebugString() returns with
        // quotation marks, e.g '"real_value"', which we don't want.
        if (field.type_kind() == zetasql::TYPE_STRING) {
          slice_value = field.string_value();
        } else if (field.type_kind() == zetasql::TYPE_BYTES) {
          slice_value = field.bytes_value();
        } else {
          slice_value = field.ShortDebugString();
        }
        slices.emplace_back(std::move(slice_key), std::move(slice_value));
      }
      row.push_back(std::move(slices));
    }
    result->push_back(std::move(row));
  }
}

// Validate the slice sql query.
absl::Status ValidateSliceSqlQuery(const zetasql::PreparedQuery& query,
                                   const std::string& sql) {
  if (query.num_columns() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid SQL statement: ", sql,
                     " Only one column should be returned."));
  }
  if (!query.column_type(0)->IsArray() ||
      !query.column_type(0)->AsArray()->element_type()->IsStruct()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid SQL statement: ", sql,
        " The each row of query result should in an Array of Struct type."));
  }
  for (const auto& field :
       query.column_type(0)->AsArray()->element_type()->AsStruct()->fields()) {
    if (!field.type->IsString() && !field.type->IsNumerical() &&
        !field.type->IsBytes()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid SQL statement: ", sql,
          " slices values must have primitive types. Found: ", field.name, ": ",
          field.type->ShortTypeName(zetasql::ProductMode::PRODUCT_INTERNAL)));
    }
  }
  return absl::OkStatus();
}

// This class converts the apache arrow array type to the corresponding
// zetasql type.
class ZetaSQLTypeVisitor : public arrow::TypeVisitor {
 public:
  ZetaSQLTypeVisitor() : zetasql_type_(nullptr) {}

  const zetasql::Type* ZetaSqlType() { return zetasql_type_; }

  arrow::Status Visit(const arrow::Int32Type& type) {
    zetasql_type_ = zetasql::types::Int32Type();
    return arrow::Status::OK();
  }

  arrow::Status Visit(const arrow::Int64Type& type) {
    zetasql_type_ = zetasql::types::Int64Type();
    return arrow::Status::OK();
  }

  arrow::Status Visit(const arrow::FloatType& type) {
    zetasql_type_ = zetasql::types::FloatType();
    return arrow::Status::OK();
  }

  arrow::Status Visit(const arrow::DoubleType& type) {
    zetasql_type_ = zetasql::types::DoubleType();
    return arrow::Status::OK();
  }

  arrow::Status Visit(const arrow::StringType& type) {
    zetasql_type_ = zetasql::types::StringType();
    return arrow::Status::OK();
  }

  arrow::Status Visit(const arrow::LargeStringType& type) {
    zetasql_type_ = zetasql::types::StringType();
    return arrow::Status::OK();
  }

  arrow::Status Visit(const arrow::BinaryType& type) {
    zetasql_type_ = zetasql::types::BytesType();
    return arrow::Status::OK();
  }

  arrow::Status Visit(const arrow::LargeBinaryType& type) {
    zetasql_type_ = zetasql::types::BytesType();
    return arrow::Status::OK();
  }

  arrow::Status Visit(const arrow::ListType& type) { return VisitList(type); }
  arrow::Status Visit(const arrow::LargeListType& type) {
    return VisitList(type);
  }

 private:
  // The life cycle of zetasql type is managed by TypeFactory.
  const zetasql::Type* zetasql_type_;

  template <class ListLikeType>
  arrow::Status VisitList(const ListLikeType& type) {
    ZetaSQLTypeVisitor visitor;
    ARROW_RETURN_NOT_OK(type.value_type()->Accept(&visitor));
    const zetasql::Type* child_zetasql_type = visitor.ZetaSqlType();
    zetasql_type_ = zetasql::types::ArrayTypeFromSimpleTypeKind(
        child_zetasql_type->kind());
    if (zetasql_type_ == nullptr) {
      return arrow::Status::TypeError(
          "Unsupported arrow data type: ", type.ToString(),
          " For ListType arrow array, we only support an array of  a primary "
          "type. A ListType of ListType is not supported currently.");
    }
    return arrow::Status::OK();
  }
};

// This class converts the apache arrow array value to the corresponding
// zetasql value.
class ZetaSQLValueVisitor : public arrow::ArrayVisitor {
 public:
  ZetaSQLValueVisitor(const zetasql::Type* zetasql_type)
      : zetasql_type_(zetasql_type), index_(0) {}

  void SetIndex(int64_t index) { index_ = index; }

  const zetasql::Value& ZetaSqlValue() const { return zetasql_value_; }

  arrow::Status Visit(const arrow::Int32Array& array) override {
    if (array.IsNull(index_)) {
      zetasql_value_ = zetasql::Value::NullInt32();
    } else {
      zetasql_value_ = zetasql::Value::Int32(array.Value(index_));
    }
    return arrow::Status::OK();
  }

  arrow::Status Visit(const arrow::Int64Array& array) override {
    if (array.IsNull(index_)) {
      zetasql_value_ = zetasql::Value::NullInt64();
    } else {
      zetasql_value_ = zetasql::Value::Int64(array.Value(index_));
    }
    return arrow::Status::OK();
  }

  arrow::Status Visit(const arrow::FloatArray& array) override {
    if (array.IsNull(index_)) {
      zetasql_value_ = zetasql::Value::NullFloat();
    } else {
      zetasql_value_ = zetasql::Value::Float(array.Value(index_));
    }
    return arrow::Status::OK();
  }

  arrow::Status Visit(const arrow::DoubleArray& array) override {
    if (array.IsNull(index_)) {
      zetasql_value_ = zetasql::Value::NullDouble();
    } else {
      zetasql_value_ = zetasql::Value::Double(array.Value(index_));
    }
    return arrow::Status::OK();
  }

  arrow::Status Visit(const arrow::StringArray& array) override {
    if (array.IsNull(index_)) {
      zetasql_value_ = zetasql::Value::NullString();
    } else {
      zetasql_value_ = zetasql::Value::StringValue(array.GetString(index_));
    }
    return arrow::Status::OK();
  }

  arrow::Status Visit(const arrow::LargeStringArray& array) override {
    if (array.IsNull(index_)) {
      zetasql_value_ = zetasql::Value::NullString();
    } else {
      zetasql_value_ = zetasql::Value::StringValue(array.GetString(index_));
    }
    return arrow::Status::OK();
  }

  arrow::Status Visit(const arrow::BinaryArray& array) override {
    if (array.IsNull(index_)) {
      zetasql_value_ = zetasql::Value::NullBytes();
    } else {
      zetasql_value_ = zetasql::Value::Bytes(array.GetString(index_));
    }
    return arrow::Status::OK();
  }

  arrow::Status Visit(const arrow::LargeBinaryArray& array) override {
    if (array.IsNull(index_)) {
      zetasql_value_ = zetasql::Value::NullBytes();
    } else {
      zetasql_value_ = zetasql::Value::Bytes(array.GetString(index_));
    }
    return arrow::Status::OK();
  }

  arrow::Status Visit(const arrow::ListArray& array) override {
    return VisitList(array);
  }

  arrow::Status Visit(const arrow::LargeListArray& array) override {
    return VisitList(array);
  }

 private:
  zetasql::Value zetasql_value_;
  const zetasql::Type* zetasql_type_;
  int64_t index_;

  template <class ListLikeType>
  arrow::Status VisitList(const ListLikeType& array) {
    if (array.IsNull(index_)) {
      zetasql_value_ = zetasql::Value::Null(zetasql_type_);
      return arrow::Status::OK();
    }

    const auto value_position = array.value_offset(index_);
    const auto value_length = array.value_length(index_);

    std::shared_ptr<arrow::Array> child_array = array.values();

    // Recursively visit child element to get child zetasql values.
    std::vector<zetasql::Value> child_zetasql_values;
    child_zetasql_values.reserve(value_length);
    const zetasql::Type* child_type =
        zetasql_type_->AsArray()->element_type();
    ZetaSQLValueVisitor child_arrow_visitor(child_type);
    for (int j = value_position; j < value_position + value_length; j++) {
      child_arrow_visitor.SetIndex(j);
      ARROW_RETURN_NOT_OK(child_array->Accept(&child_arrow_visitor));
      const zetasql::Value& value = child_arrow_visitor.ZetaSqlValue();
      child_zetasql_values.push_back(value);
    }

    // Generate current element zetasql value based on child zetasql value
    // and type.
    zetasql_value_ = zetasql::Value::UnsafeArray(
        zetasql::types::ArrayTypeFromSimpleTypeKind(child_type->kind()),
        std::move(child_zetasql_values));
    return arrow::Status::OK();
  }
};

// An iterator that wraps the RecordBatch data structure.
class RecordBatchEvaluatorTableIterator
    : public zetasql::EvaluatorTableIterator {
 public:
  RecordBatchEvaluatorTableIterator(const RecordBatchEvaluatorTableIterator&) =
      delete;
  RecordBatchEvaluatorTableIterator& operator=(
      const RecordBatchEvaluatorTableIterator&) = delete;

  RecordBatchEvaluatorTableIterator(
      const arrow::RecordBatch& record_batch,
      const std::vector<SimpleTable::NameAndType>& columns_name_and_type)
      : record_batch_(record_batch),
        columns_name_and_type_(columns_name_and_type),
        current_row_index_(-1),
        cancelled_(false),
        status_(absl::OkStatus()),
        zetasql_value_visitors_(columns_name_and_type.size()) {
    for (int i = 0; i < zetasql_value_visitors_.size(); i++) {
      zetasql_value_visitors_[i] = absl::make_unique<ZetaSQLValueVisitor>(
          columns_name_and_type[i].second);
    }

    // Build a map: "record batch colume name" -> "record batch_ colume index".
    std::unordered_map<std::string, int> record_batch_column_name_to_index;
    for (int i = 0; i < record_batch_.num_columns(); i++) {
      record_batch_column_name_to_index[record_batch_.column_name(i)] = i;
    }

    // Build a map: "virtual sql table colume index" -> "record batch colume
    // index".
    for (int zetasql_index = 0;
         zetasql_index < columns_name_and_type_.size(); zetasql_index++) {
      const std::string& column_name =
          columns_name_and_type_[zetasql_index].first;
      index_map_[zetasql_index] =
          record_batch_column_name_to_index[column_name];
    }
  }

  int NumColumns() const override { return columns_name_and_type_.size(); }

  std::string GetColumnName(int i) const override {
    return columns_name_and_type_[i].first;
  }

  const zetasql::Type* GetColumnType(int i) const override {
    return columns_name_and_type_[i].second;
  }

  bool NextRow() override {
    if (cancelled_) {
      return false;
    }
    if (++current_row_index_ >= record_batch_.num_rows()) {
      return false;
    }
    return true;
  }

  const zetasql::Value& GetValue(int sql_column_index) const override {
    // Convert to real record batch column index.
    int record_batch_column_index = index_map_.at(sql_column_index);
    std::shared_ptr<arrow::Array> column_array =
        record_batch_.column(record_batch_column_index);

    const std::unique_ptr<ZetaSQLValueVisitor>& zetasql_value_visitor =
        zetasql_value_visitors_.at(sql_column_index);
    zetasql_value_visitor->SetIndex(current_row_index_);
    arrow::Status status = column_array->Accept(zetasql_value_visitor.get());
    assert(status.ok());

    // A reference is returned here. So zetasql_arrow_visitor must live
    // longer than current function scope.
    return zetasql_value_visitor->ZetaSqlValue();
  }

  absl::Status Status() const final {
    if (cancelled_) {
      return absl::CancelledError(
          "RecordBatchEvaluatorTableIterator was cancelled");
    }
    return absl::OkStatus();
  }

  absl::Status Cancel() final {
    cancelled_ = true;
    return absl::OkStatus();
  }

 private:
  const arrow::RecordBatch& record_batch_;
  const std::vector<SimpleTable::NameAndType>& columns_name_and_type_;
  int current_row_index_;
  bool cancelled_;
  absl::Status status_;

  std::vector<std::unique_ptr<ZetaSQLValueVisitor>> zetasql_value_visitors_;
  // Map the index from zetasql table to record batch table. We need this
  // mapping because the some column format of the record batch table is not
  // supported by the zetasql.
  std::unordered_map<int, int> index_map_;
};
}  // namespace

// static. Create RecordBatchSQLSliceQuery.
absl::Status RecordBatchSQLSliceQuery::Make(
    const std::string& sql, std::shared_ptr<arrow::Schema> arrow_schema,
    std::unique_ptr<RecordBatchSQLSliceQuery>* result) {
  // Build name and type per column.
  std::vector<zetasql::SimpleTable::NameAndType> columns_name_and_type;
  for (int i = 0; i < arrow_schema->num_fields(); i++) {
    ZetaSQLTypeVisitor type_visitor;
    arrow::Status status =
        arrow_schema->field(i)->type()->Accept(&type_visitor);
    // Only add supported columns to the sql table.
    if (status.ok()) {
      columns_name_and_type.emplace_back(arrow_schema->field(i)->name(),
                                         type_visitor.ZetaSqlType());
    }
  }

  // Build sql table.
  std::unique_ptr<SimpleTable> table =
      absl::make_unique<SimpleTable>(kTableName, columns_name_and_type);

  // Build sql Catalog.
  std::unique_ptr<zetasql::SimpleCatalog> catalog =
      std::make_unique<SimpleCatalog>("catalog");
  catalog->AddZetaSQLFunctions();
  catalog->AddTable(table->Name(), table.get());

  // Prepare the query.
  const zetasql::EvaluatorOptions evaluator_options;
  std::unique_ptr<zetasql::PreparedQuery> query =
      absl::make_unique<zetasql::PreparedQuery>(sql, evaluator_options);
  zetasql::AnalyzerOptions analyzer_options;

  absl::Status status = query->Prepare(analyzer_options, catalog.get());
  if (absl::IsInvalidArgument(status) &&
      absl::StartsWith(status.message(), "Unrecognized name:")) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unable to analyze SQL query. Error: ",
        absl::StatusCodeToString(status.code()), " : ", status.message(),
        ". Are you querying any unsupported column?"));
  }
  TFX_BSL_RETURN_IF_ERROR(status);

  TFX_BSL_RETURN_IF_ERROR(ValidateSliceSqlQuery(*query, sql));
  *result = absl::WrapUnique(new RecordBatchSQLSliceQuery(
      std::move(arrow_schema), std::move(columns_name_and_type),
      std::move(table), std::move(catalog), std::move(query)));
  return absl::OkStatus();
}

RecordBatchSQLSliceQuery::RecordBatchSQLSliceQuery(
    std::shared_ptr<arrow::Schema> arrow_schema,
    std::vector<zetasql::SimpleTable::NameAndType> columns_name_and_type,
    std::unique_ptr<SimpleTable> table,
    std::unique_ptr<zetasql::SimpleCatalog> catalog,
    std::unique_ptr<zetasql::PreparedQuery> query)
    : arrow_schema_(std::move(arrow_schema)),
      columns_name_and_type_(std::move(columns_name_and_type)),
      table_(std::move(table)),
      catalog_(std::move(catalog)),
      query_(std::move(query)) {}

absl::Status RecordBatchSQLSliceQuery::Execute(
    const arrow::RecordBatch& record_batch,
    std::vector<std::vector<std::vector<std::pair<std::string, std::string>>>>*
        result) {
  // Check if the schema is as same as the stored schema.
  if (!record_batch.schema()->Equals(*arrow_schema_)) {
    return absl::InvalidArgumentError("Unexpected RecordBatch schema.");
  }

  // Add record batch to the table.
  table_->SetEvaluatorTableIteratorFactory(
      [&record_batch, this](absl::Span<const int> columns)
          -> zetasql_base::StatusOr<
              std::unique_ptr<zetasql::EvaluatorTableIterator>> {
        return absl::make_unique<RecordBatchEvaluatorTableIterator>(
            record_batch, this->columns_name_and_type_);
      });

  // Excute.
  zetasql_base::StatusOr<std::unique_ptr<zetasql::EvaluatorTableIterator>>
      query_result_iterator = query_->Execute();
  TFX_BSL_RETURN_IF_ERROR(query_result_iterator.status());

  // Convert the query result to the output.
  ConvertQueryResultToSlices(std::move(query_result_iterator.value()), result);
  return absl::OkStatus();
}

RecordBatchSQLSliceQuery::~RecordBatchSQLSliceQuery() {}

}  // namespace tfx_bsl
