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
#ifndef THIRD_PARTY_PY_TFX_BSL_GOOGLE_CC_ARROW_SQL_UTIL_H_
#define THIRD_PARTY_PY_TFX_BSL_GOOGLE_CC_ARROW_SQL_UTIL_H_

#include <memory>
#include <vector>

#include "zetasql/public/evaluator.h"
#include "zetasql/public/evaluator_table_iterator.h"
#include "zetasql/public/simple_catalog.h"
#include "tfx_bsl/cc/util/status.h"
#include "pybind11/pybind11.h"

namespace arrow {
class Array;
class RecordBatch;
class Schema;
}  // namespace arrow

namespace tfx_bsl {

class RecordBatchSQLSliceQuery {
 public:
  static Status Make(const std::string& sql,
                     std::shared_ptr<arrow::Schema> arrow_schema,
                     std::unique_ptr<RecordBatchSQLSliceQuery>* result);
  ~RecordBatchSQLSliceQuery();

  // Creates slice keys for each row in a RecordBatch, according to a SQL
  // statement.
  Status Execute(
      const arrow::RecordBatch& record_batch,
      std::vector<
          std::vector<std::vector<std::pair<std::string, std::string>>>>*
          result);

  RecordBatchSQLSliceQuery(const RecordBatchSQLSliceQuery&) = delete;
  RecordBatchSQLSliceQuery& operator=(const RecordBatchSQLSliceQuery&) = delete;

 private:
  std::shared_ptr<arrow::Schema> arrow_schema_;
  std::vector<zetasql::SimpleTable::NameAndType> columns_name_and_type_;
  std::unique_ptr<zetasql::SimpleTable> table_;
  std::unique_ptr<zetasql::SimpleCatalog> catalog_;
  std::unique_ptr<zetasql::PreparedQuery> query_;

  RecordBatchSQLSliceQuery(
      std::shared_ptr<arrow::Schema> arrow_schema,
      std::vector<zetasql::SimpleTable::NameAndType> columns_name_and_type,
      std::unique_ptr<zetasql::SimpleTable> table,
      std::unique_ptr<zetasql::SimpleCatalog> catalog,
      std::unique_ptr<zetasql::PreparedQuery> query);
};

}  // namespace tfx_bsl

#endif  // THIRD_PARTY_PY_TFX_BSL_GOOGLE_CC_ARROW_SQL_UTIL_H_
