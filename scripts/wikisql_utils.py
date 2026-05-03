"""
WikiSQL utilities for text-to-SQL RLVR experiments.

Provides:
- Dataset loading from local JSONL files (downloaded from GitHub)
- Prompt construction with table schema
- SQLite execution for reward computation
- Reward scoring based on execution correctness

WikiSQL uses symbolic column names (col0, col1, col2...) for SQL execution.
The model must learn to map natural language to these symbolic columns.

Data files should be in data/data/ directory:
- train.jsonl, dev.jsonl, test.jsonl (questions + SQL)
- train.tables.jsonl, dev.tables.jsonl, test.tables.jsonl (table schemas + rows)

Download with:
    cd data && curl -L -o wikisql.tar.bz2 "https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"
    tar -xjf wikisql.tar.bz2
"""

import json
import os
import random
import re
import sqlite3
from dataclasses import dataclass
from typing import Optional


AGG_OPERATORS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPERATORS = ["=", ">", "<", "OP"]

# Data directory relative to this script
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/data")


@dataclass
class WikiSQLExample:
    """A single WikiSQL example with prompt, gold SQL, and table info."""
    prompt: str
    question: str
    gold_sql: str
    table_id: str
    header: list[str]
    types: list[str]
    rows: list[list]


# Table name for SQL (avoid reserved keyword 'table')
TABLE_NAME = "t"


def build_gold_sql(sql_dict: dict) -> str:
    """
    Reconstruct SQL query string from WikiSQL structured representation.

    Args:
        sql_dict: Dictionary with 'sel', 'agg', 'conds' fields
                  conds is a list of [col_idx, op_idx, value] tuples

    Returns:
        SQL query string using col0, col1, col2... column names
    """
    sel_col = sql_dict["sel"]
    agg_idx = sql_dict["agg"]

    # SELECT clause
    if agg_idx == 0:
        select_clause = f"col{sel_col}"
    else:
        select_clause = f"{AGG_OPERATORS[agg_idx]}(col{sel_col})"

    sql = f"SELECT {select_clause} FROM {TABLE_NAME}"

    # WHERE clause - conds is list of [col_idx, op_idx, value]
    conds = sql_dict.get("conds", [])

    if conds:
        where_parts = []
        for cond in conds:
            col_idx, op_idx, val = cond[0], cond[1], cond[2]
            op = COND_OPERATORS[op_idx]
            # Quote string values, escape single quotes
            if isinstance(val, str):
                val_escaped = val.replace("'", "''")
                val_str = f"'{val_escaped}'"
            else:
                val_str = str(val)
            where_parts.append(f"col{col_idx} {op} {val_str}")

        sql += " WHERE " + " AND ".join(where_parts)

    return sql


def build_prompt(question: str, header: list[str], table_id: str = "",
                 few_shot: bool = True, show_col_mapping: bool = True) -> str:
    """
    Build prompt for text-to-SQL generation.

    Args:
        question: Natural language question
        header: List of column names from the original table
        table_id: Optional table identifier
        few_shot: Whether to include a few-shot example
        show_col_mapping: Whether to show col0->header[0] mapping

    Returns:
        Formatted prompt string
    """
    # Build column schema with symbolic names
    if show_col_mapping:
        col_parts = [f"col{i}:{name}" for i, name in enumerate(header)]
    else:
        col_parts = [f"col{i}" for i in range(len(header))]
    schema_str = ", ".join(col_parts)

    prefix = (
        "You are a SQL assistant. Generate a SQL query for the question.\n"
        f"Use symbolic column names (col0, col1, ...) and table name '{TABLE_NAME}'.\n"
        "Return only the SQL query, nothing else.\n"
    )

    if few_shot:
        prefix += (
            "\nExample:\n"
            "Columns: col0:Name, col1:Age, col2:City\n"
            "Question: What is the age of Alice?\n"
            f"SQL: SELECT col1 FROM {TABLE_NAME} WHERE col0 = 'Alice'\n"
        )

    prompt = f"{prefix}\nColumns: {schema_str}\nQuestion: {question}\nSQL:"
    return prompt


def create_sqlite_table(conn: sqlite3.Connection, header: list[str],
                        rows: list[list], table_name: str = TABLE_NAME) -> None:
    """
    Create an in-memory SQLite table from WikiSQL data.

    Args:
        conn: SQLite connection
        header: Column names (used only for count, columns are col0, col1, ...)
        rows: 2D list of row data
        table_name: Name for the table (default: TABLE_NAME)
    """
    cursor = conn.cursor()

    # Drop existing table if any
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

    # Create table with TEXT columns (WikiSQL types are mostly "text")
    col_defs = ", ".join([f"col{i} TEXT" for i in range(len(header))])
    cursor.execute(f"CREATE TABLE {table_name} ({col_defs})")

    # Insert rows
    if rows:
        placeholders = ", ".join(["?" for _ in range(len(header))])
        for row in rows:
            # Ensure row has correct number of columns
            padded_row = list(row) + [""] * (len(header) - len(row))
            padded_row = padded_row[:len(header)]
            cursor.execute(f"INSERT INTO {table_name} VALUES ({placeholders})", padded_row)

    conn.commit()


def execute_sql(conn: sqlite3.Connection, sql: str) -> tuple[bool, Optional[list], Optional[str]]:
    """
    Execute SQL query and return results.

    Args:
        conn: SQLite connection with table already created
        sql: SQL query string

    Returns:
        Tuple of (success, results, error_message)
        - success: True if query executed without error
        - results: List of result rows, or None if error
        - error_message: Error string if failed, None otherwise
    """
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        return True, results, None
    except sqlite3.Error as e:
        return False, None, str(e)


def normalize_result(results: Optional[list]) -> list:
    """
    Normalize SQL results for comparison.

    Handles:
    - None/empty results
    - Type conversions (int strings, floats)
    - Case normalization for strings
    """
    if not results:
        return []

    normalized = []
    for row in results:
        norm_row = []
        for val in row:
            if val is None:
                norm_row.append(None)
            elif isinstance(val, (int, float)):
                # Keep numbers as-is
                norm_row.append(val)
            else:
                # Normalize strings: strip, lowercase
                s = str(val).strip().lower()
                # Try to parse as number
                try:
                    if "." in s:
                        norm_row.append(float(s))
                    else:
                        norm_row.append(int(s))
                except ValueError:
                    norm_row.append(s)
        normalized.append(tuple(norm_row))

    return normalized


def normalize_table_name(sql: str) -> str:
    """
    Normalize table name in SQL to use TABLE_NAME.

    Handles 'table', 't', or quoted variants.
    """
    # Replace FROM table/FROM "table"/FROM 'table' with FROM t
    sql = re.sub(r"\bFROM\s+['\"]?table['\"]?\b", f"FROM {TABLE_NAME}", sql, flags=re.IGNORECASE)
    return sql


def extract_sql_from_response(response: str) -> str:
    """
    Extract SQL query from model response.

    Handles:
    - Response with just SQL
    - Response with SQL: prefix
    - Response with extra text around SQL
    - Normalizes table name to TABLE_NAME
    """
    response = response.strip()

    # Remove common prefixes
    for prefix in ["SQL:", "sql:", "Query:", "query:", "Answer:", "answer:"]:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()

    # Try to find SELECT statement
    match = re.search(r"(SELECT\s+.+?)(?:;|$)", response, re.IGNORECASE | re.DOTALL)
    if match:
        sql = match.group(1).strip()
        # Remove trailing semicolon if present
        if sql.endswith(";"):
            sql = sql[:-1]
        # Normalize table name
        sql = normalize_table_name(sql)
        return sql

    # If no SELECT found, return cleaned response
    # Remove trailing semicolon
    if response.endswith(";"):
        response = response[:-1]

    # Normalize table name
    response = normalize_table_name(response)
    return response


def is_valid_sql(sql: str) -> bool:
    """Check if SQL string has basic valid structure."""
    sql_upper = sql.upper().strip()
    return sql_upper.startswith("SELECT") and "FROM" in sql_upper


def score_sql_execution(
    generated_sql: str,
    gold_sql: str,
    header: list[str],
    rows: list[list],
) -> tuple[float, dict]:
    """
    Score generated SQL by comparing execution results.

    Reward structure:
    - 1.0: Execution result matches gold SQL result
    - 0.5: Valid SQL, executes, but wrong result
    - 0.1: Valid SQL syntax but execution error (e.g., wrong column)
    - 0.0: Invalid SQL (parse error)

    Args:
        generated_sql: SQL query generated by the model
        gold_sql: Ground truth SQL query
        header: Table column headers
        rows: Table row data

    Returns:
        Tuple of (reward, details_dict)
    """
    details = {
        "valid_sql": False,
        "executes": False,
        "correct_result": False,
        "generated_sql": generated_sql,
        "gold_sql": gold_sql,
        "error": None,
        "generated_result": None,
        "gold_result": None,
    }

    # Extract SQL from response
    generated_sql = extract_sql_from_response(generated_sql)
    details["generated_sql"] = generated_sql

    # Check basic SQL validity
    if not is_valid_sql(generated_sql):
        details["error"] = "Invalid SQL structure"
        return 0.0, details

    details["valid_sql"] = True

    # Create in-memory database
    conn = sqlite3.connect(":memory:")
    try:
        create_sqlite_table(conn, header, rows)

        # Execute gold SQL first
        gold_success, gold_results, gold_error = execute_sql(conn, gold_sql)
        if not gold_success:
            # This shouldn't happen with valid WikiSQL data
            details["error"] = f"Gold SQL error: {gold_error}"
            conn.close()
            return 0.1, details

        details["gold_result"] = gold_results

        # Execute generated SQL
        gen_success, gen_results, gen_error = execute_sql(conn, generated_sql)

        if not gen_success:
            details["error"] = f"Execution error: {gen_error}"
            conn.close()
            return 0.1, details

        details["executes"] = True
        details["generated_result"] = gen_results

        # Compare results
        gold_normalized = normalize_result(gold_results)
        gen_normalized = normalize_result(gen_results)

        if gold_normalized == gen_normalized:
            details["correct_result"] = True
            conn.close()
            return 1.0, details
        else:
            conn.close()
            return 0.5, details

    except Exception as e:
        details["error"] = f"Unexpected error: {str(e)}"
        conn.close()
        return 0.0, details


def check_data_exists(data_dir: str = DATA_DIR) -> bool:
    """Check if WikiSQL data files exist."""
    required_files = ["train.jsonl", "train.tables.jsonl"]
    return all(os.path.exists(os.path.join(data_dir, f)) for f in required_files)


def load_tables(split: str, data_dir: str = DATA_DIR) -> dict:
    """
    Load table data from tables JSONL file.

    Args:
        split: Dataset split ("train", "dev", "test")
        data_dir: Directory containing data files

    Returns:
        Dictionary mapping table_id -> table dict
    """
    if not check_data_exists(data_dir):
        raise FileNotFoundError(
            f"WikiSQL data not found in {data_dir}.\n"
            "Run: bash scripts/download_wikisql.sh"
        )

    # Map split names
    file_split = "dev" if split == "validation" else split
    tables_file = os.path.join(data_dir, f"{file_split}.tables.jsonl")

    tables = {}
    with open(tables_file, "r", encoding="utf-8") as f:
        for line in f:
            table = json.loads(line.strip())
            tables[table["id"]] = table

    return tables


def load_examples(split: str, data_dir: str = DATA_DIR) -> list[dict]:
    """
    Load question/SQL examples from JSONL file.

    Args:
        split: Dataset split ("train", "dev", "test")
        data_dir: Directory containing data files

    Returns:
        List of example dictionaries
    """
    # Map split names
    file_split = "dev" if split == "validation" else split
    examples_file = os.path.join(data_dir, f"{file_split}.jsonl")

    examples = []
    with open(examples_file, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line.strip()))

    return examples


class WikiSQLDataset:
    """
    WikiSQL dataset wrapper for RLVR experiments.

    Usage:
        dataset = WikiSQLDataset(split="train", max_examples=1000)
        example = dataset.sample()
        reward, details = dataset.score(example, generated_sql)
    """

    def __init__(
        self,
        split: str = "train",
        max_examples: Optional[int] = None,
        few_shot: bool = True,
        show_col_mapping: bool = True,
        seed: Optional[int] = None,
        data_dir: str = DATA_DIR,
    ):
        """
        Initialize WikiSQL dataset.

        Args:
            split: Dataset split ("train", "validation"/"dev", "test")
            max_examples: Maximum number of examples to load (None for all)
            few_shot: Include few-shot example in prompts
            show_col_mapping: Show col0:ColumnName mapping in prompts
            seed: Random seed for sampling
            data_dir: Directory containing WikiSQL data files
        """
        self.split = split
        self.few_shot = few_shot
        self.show_col_mapping = show_col_mapping
        self.data_dir = data_dir

        # Load tables and examples
        print(f"Loading WikiSQL tables from {data_dir}...")
        self.tables = load_tables(split, data_dir)
        print(f"Loaded {len(self.tables)} tables")

        print(f"Loading WikiSQL examples...")
        raw_examples = load_examples(split, data_dir)
        print(f"Loaded {len(raw_examples)} examples")

        # Subsample if needed
        if max_examples is not None and max_examples < len(raw_examples):
            if seed is not None:
                random.seed(seed)
            raw_examples = random.sample(raw_examples, max_examples)

        self.examples = self._build_examples(raw_examples)

        if seed is not None:
            random.seed(seed)

    def _build_examples(self, raw_examples: list[dict]) -> list[WikiSQLExample]:
        """Convert raw examples to WikiSQLExample objects."""
        examples = []
        skipped = 0

        for item in raw_examples:
            question = item["question"]
            table_id = item["table_id"]
            sql_dict = item["sql"]

            # Look up table
            if table_id not in self.tables:
                skipped += 1
                continue

            table = self.tables[table_id]
            header = table["header"]
            types = table["types"]
            rows = table["rows"]

            gold_sql = build_gold_sql(sql_dict)
            prompt = build_prompt(
                question=question,
                header=header,
                table_id=table_id,
                few_shot=self.few_shot,
                show_col_mapping=self.show_col_mapping,
            )

            examples.append(WikiSQLExample(
                prompt=prompt,
                question=question,
                gold_sql=gold_sql,
                table_id=table_id,
                header=header,
                types=types,
                rows=rows,
            ))

        if skipped > 0:
            print(f"Warning: skipped {skipped} examples with missing tables")

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> WikiSQLExample:
        return self.examples[idx]

    def sample(self) -> WikiSQLExample:
        """Sample a random example."""
        return random.choice(self.examples)

    def score(self, example: WikiSQLExample, generated_response: str) -> tuple[float, dict]:
        """
        Score a generated SQL response.

        Args:
            example: The WikiSQLExample used for generation
            generated_response: Model's generated text

        Returns:
            Tuple of (reward, details_dict)
        """
        return score_sql_execution(
            generated_sql=generated_response,
            gold_sql=example.gold_sql,
            header=example.header,
            rows=example.rows,
        )

    def build_eval_set(self, num_examples: int, seed: int) -> list[WikiSQLExample]:
        """
        Create a fixed evaluation set.

        Args:
            num_examples: Number of examples in eval set
            seed: Random seed for reproducibility

        Returns:
            List of WikiSQLExample objects
        """
        rng_state = random.getstate()
        random.seed(seed)

        indices = random.sample(range(len(self.examples)), min(num_examples, len(self.examples)))
        eval_examples = [self.examples[i] for i in indices]

        random.setstate(rng_state)
        return eval_examples


def test_wikisql_utils():
    """Simple test to verify utilities work."""
    print("Loading WikiSQL dataset (100 examples)...")
    dataset = WikiSQLDataset(split="train", max_examples=100, seed=42)
    print(f"Loaded {len(dataset)} examples")

    # Test a sample
    example = dataset[0]
    print(f"\n--- Example ---")
    print(f"Question: {example.question}")
    print(f"Gold SQL: {example.gold_sql}")
    print(f"Columns: {example.header}")
    print(f"Num rows: {len(example.rows)}")
    print(f"\nPrompt:\n{example.prompt[:500]}...")

    # Test scoring with gold SQL (should be 1.0)
    reward, details = dataset.score(example, example.gold_sql)
    print(f"\nScoring gold SQL: reward={reward}")
    assert reward == 1.0, f"Expected 1.0 for gold SQL, got {reward}"

    # Test scoring with wrong SQL
    wrong_sql = "SELECT col0 FROM table"
    reward, details = dataset.score(example, wrong_sql)
    print(f"Scoring wrong SQL: reward={reward}")

    # Test scoring with invalid SQL
    invalid_sql = "This is not SQL"
    reward, details = dataset.score(example, invalid_sql)
    print(f"Scoring invalid SQL: reward={reward}")
    assert reward == 0.0, f"Expected 0.0 for invalid SQL, got {reward}"

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_wikisql_utils()
