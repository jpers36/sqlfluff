""" The MSSQL T-SQL dialect.

https://docs.microsoft.com/en-us/sql/t-sql/language-elements/language-elements-transact-sql


"""
from enum import Enum
from typing import Any, Generator, List, Tuple, NamedTuple, Optional, Union

from sqlfluff.core.parser import (
    Matchable,

    BaseSegment,
    KeywordSegment,
    
    Sequence,
    GreedyUntil,
    StartsWith,
    OneOf,
    Delimited,
   
    Bracketed,
    AnyNumberOf,
    Ref,
    SegmentGenerator,
    Anything,
    Indent,
    Dedent,
    Nothing,
    OptionallyBracketed,

    RegexLexer,
    CodeSegment,
    CommentSegment,
    StringParser,
   
    RegexParser,
    Conditional,
)

from sqlfluff.core.dialects.base import Dialect
from sqlfluff.core.dialects.common import AliasInfo
from sqlfluff.core.parser.segments.base import BracketedSegment

from sqlfluff.dialects.tsql_keywords import (
    RESERVED_KEYWORDS,
    UNRESERVED_KEYWORDS,
)

from sqlfluff.core.dialects import load_raw_dialect
from sqlfluff.dialects.dialect_ansi import StatementSegment

ansi_dialect = load_raw_dialect("ansi")
tsql_dialect = ansi_dialect.copy_as("tsql")

# Clear ANSI Keywords and add all TSQL keywords
tsql_dialect.sets("unreserved_keywords").clear()
tsql_dialect.sets("unreserved_keywords").update(UNRESERVED_KEYWORDS)
tsql_dialect.sets("reserved_keywords").clear()
tsql_dialect.sets("reserved_keywords").update(RESERVED_KEYWORDS)

tsql_dialect.replace(
    ParameterNameSegment=RegexParser(
        r"[@][A-Za-z0-9_]+", CodeSegment, name="parameter", type="parameter"
    ),
    QuotedIdentifierSegment=Bracketed(
        RegexParser(
            r"[A-Z][A-Z0-9_]*", CodeSegment, name="quoted_identifier", type="identifier"
        ),
        bracket_type="square",
    ),
    NakedIdentifierSegment=SegmentGenerator(
        # Generate the anti template from the set of reserved keywords
        lambda dialect: RegexParser(
            r"[#A-Z0-9_]*[A-Z][A-Z0-9_]*",
            CodeSegment,
            name="naked_identifier",
            type="identifier",
            anti_template=r"^(" + r"|".join(dialect.sets("reserved_keywords")) + r")$",
        )
    ),
    Expression_A_Grammar=Sequence(
        OneOf(
            Ref("Expression_C_Grammar"),
            Sequence(
                OneOf(
                    Ref("PositiveSegment"),
                    Ref("NegativeSegment"),
                    # Ref('TildeSegment'),
                    "NOT",
                ),
                Ref("Expression_C_Grammar"),
            ),
        ),
        AnyNumberOf(
            OneOf(
                Sequence(
                    OneOf(
                        Sequence(
                            Ref.keyword("NOT", optional=True),
                            Ref("LikeGrammar"),
                        ),
                        Sequence(
                            Ref("BinaryOperatorGrammar"),
                            Ref.keyword("NOT", optional=True),
                        ),
                        # We need to add a lot more here...
                    ),
                    Ref("Expression_C_Grammar"),
                    Sequence(
                        Ref.keyword("ESCAPE"),
                        Ref("Expression_C_Grammar"),
                        optional=True,
                    ),
                ),
                Sequence(
                    Ref.keyword("NOT", optional=True),
                    "IN",
                    Bracketed(
                        OneOf(
                            Delimited(
                                Ref("LiteralGrammar"),
                                Ref("IntervalExpressionSegment"),
                            ),
                            Ref("SelectableGrammar"),
                            ephemeral_name="InExpression",
                        )
                    ),
                ),
                Sequence(
                    Ref.keyword("NOT", optional=True),
                    "IN",
                    Ref("FunctionSegment"),  # E.g. UNNEST()
                ),
                Sequence(
                    "IS",
                    Ref.keyword("NOT", optional=True),
                    Ref("IsClauseGrammar"),
                ),
                Sequence(
                    # e.g. NOT EXISTS, but other expressions could be met as
                    # well by inverting the condition with the NOT operator
                    "NOT",
                    Ref("Expression_C_Grammar"),
                ),
                Sequence(
                    Ref.keyword("NOT", optional=True),
                    "BETWEEN",
                    # In a between expression, we're restricted to arithmetic operations
                    # because if we look for all binary operators then we would match AND
                    # as both an operator and also as the delimiter within the BETWEEN
                    # expression.
                    Ref("Expression_C_Grammar"),
                    AnyNumberOf(
                        Sequence(
                            Ref("ArithmeticBinaryOperatorGrammar"),
                            Ref("Expression_C_Grammar"),
                        )
                    ),
                    "AND",
                    Ref("Expression_C_Grammar"),
                    AnyNumberOf(
                        Sequence(
                            Ref("ArithmeticBinaryOperatorGrammar"),
                            Ref("Expression_C_Grammar"),
                        )
                    ),
                ),
            )
        ),
    ),
    IsClauseGrammar=OneOf(
        "NULL",
        Ref("BooleanLiteralGrammar"),
    ),
    FunctionContentsGrammar=AnyNumberOf(
        Ref("ExpressionSegment"),
        # A Cast-like function
        Sequence(Ref("ExpressionSegment"), "AS", Ref("DatatypeSegment")),
        # An extract-like or substring-like function
        Sequence(
            OneOf(Ref("DatetimeUnitSegment"), Ref("ExpressionSegment")),
            "FROM",
            Ref("ExpressionSegment"),
        ),
        Sequence(
            # Allow an optional distinct keyword here.
            Ref.keyword("DISTINCT", optional=True),
            OneOf(
                # Most functions will be using the delimited route
                # but for COUNT(*) or similar we allow the star segment
                # here.
                Ref("StarSegment"),
                Delimited(Ref("FunctionContentsExpressionGrammar")),
            ),
        ),
        Ref(
            "OrderByClauseSegment"
        ),  # used by string_agg (postgres), group_concat (exasol), listagg (snowflake)...
        # like a function call: POSITION ( 'QL' IN 'SQL')
        Sequence(
            OneOf(Ref("QuotedLiteralSegment"), Ref("SingleIdentifierGrammar")),
            "IN",
            OneOf(Ref("QuotedLiteralSegment"), Ref("SingleIdentifierGrammar")),
        ),
    ),
    FromClauseTerminatorGrammar=OneOf(
        "WHERE",
        "GROUP",
        "ORDER",
        "HAVING",
        Ref("SetOperatorSegment"),
        Ref("WithNoSchemaBindingClauseSegment"),
    ),
    WhereClauseTerminatorGrammar=OneOf(
        "GROUP", "ORDER", "HAVING",
    ),
    LikeGrammar=OneOf("LIKE"),
    DateTimeLiteralGrammar=Nothing(),
    PostFunctionGrammar=Nothing(),
    TemporaryTransientGrammar=Nothing(),
)

tsql_dialect.patch_lexer_matchers(
    [
        RegexLexer(
            "inline_comment",
            r"(--)[^\n]*",
            CommentSegment,
            segment_kwargs={"trim_start": ("--")},
        ),
    ]
)

tsql_dialect.insert_lexer_matchers(
    [
        RegexLexer(
            "atsign",
            r"[@][a-zA-Z0-9_]+",
            CodeSegment,
        ),
        RegexLexer(
            "hash",
            r"[#][a-zA-Z0-9_]+",
            CodeSegment,
        ),
    ],
    before="code",
)

@tsql_dialect.segment()
class GoStatementSegment(BaseSegment):
    """This is a Go statement to signal end of batch"""
    type = "go_statement"
    type = "go_statement"
    match_grammar = Sequence("GO")

@tsql_dialect.segment()
class SchemaNameSegment(BaseSegment):
    """This is a schema name optionally bracketed"""
    type = "schema_name"
    name = "schema"
    match_grammar = Sequence(
        Ref("StartSquareBracketSegment", optional=True),
        Ref("SingleIdentifierGrammar"),
        Ref("EndSquareBracketSegment", optional=True),
        Ref("DotSegment"),
    )


@tsql_dialect.segment()
class ObjectNameSegment(BaseSegment):
    """This is the body of a `CREATE FUNCTION AS` statement."""

    type = "object_name"
    match_grammar = Sequence(
        Ref("StartSquareBracketSegment", optional=True),
        Ref("SingleIdentifierGrammar"),
        Ref("EndSquareBracketSegment", optional=True),
    )


@tsql_dialect.segment(replace=True)
class CreateTableStatementSegment(BaseSegment):
    """A `CREATE TABLE` statement."""

    type = "create_table_statement"
    # https://docs.microsoft.com/en-us/sql/t-sql/statements/create-table-transact-sql?view=sql-server-ver15

    match_grammar = Sequence(
        "CREATE",
        Ref("OrReplaceGrammar", optional=True),
        Ref("TemporaryTransientGrammar", optional=True),
        "TABLE",
        Ref("IfNotExistsGrammar", optional=True),
        Ref("SchemaNameSegment", optional=True),
        Ref("TableReferenceSegment"),
        # Anything(),
        OneOf(
            # Columns and comment syntax:
            Sequence(
                Bracketed(
                    Delimited(
                        OneOf(
                            Ref("TableConstraintSegment"),
                            Ref("ColumnDefinitionSegment"),
                        ),
                    )
                ),
                Ref("CommentClauseSegment", optional=True),
            ),
            # Create AS syntax:
            Sequence(
                "AS",
                OptionallyBracketed(Ref("SelectableGrammar")),
            ),
            # Create like syntax
            Sequence("LIKE", Ref("TableReferenceSegment")),
        ),
        Ref("GoStatementSegment", optional=True),
    )


@tsql_dialect.segment(replace=True)
class DatatypeSegment(BaseSegment):
    """A data type segment."""

    type = "data_type"
    match_grammar = Sequence(
        Sequence(
            # Some dialects allow optional qualification of data types with schemas
            Sequence(
                Ref("StartSquareBracketSegment", optional=True),
                Ref("SingleIdentifierGrammar"),
                Ref("EndSquareBracketSegment", optional=True),
                Ref("DotSegment"),
                allow_gaps=False,
                optional=True,
            ),
            Ref("StartSquareBracketSegment", optional=True),
            Ref("DatatypeIdentifierSegment"),
            Ref("EndSquareBracketSegment", optional=True),
            allow_gaps=False,
        ),
        Bracketed(
            OneOf(
                Delimited(Ref("ExpressionSegment")),
                # The brackets might be empty for some cases...
                optional=True,
            ),
            # There may be no brackets for some data types
            optional=True,
        ),
        Ref("CharCharacterSetSegment", optional=True),
    )


@tsql_dialect.segment(replace=True)
class ColumnOptionSegment(BaseSegment):
    """A column option; each CREATE TABLE column can have 0 or more."""

    type = "column_constraint"
    # Column constraint from
    # https://www.postgresql.org/docs/12/sql-createtable.html
    match_grammar = Sequence(
        Sequence(
            "CONSTRAINT",
            Ref("ObjectReferenceSegment"),  # Constraint name
            optional=True,
        ),
        OneOf(
            Sequence(Ref.keyword("NOT", optional=True), "NULL"),  # NOT NULL or NULL
            Sequence(  # DEFAULT <value>
                "DEFAULT",
                OneOf(
                    Ref("LiteralGrammar"),
                    Ref("FunctionSegment"),
                    # ?? Ref('IntervalExpressionSegment')
                ),
            ),
            Ref("PrimaryKeyGrammar"),
            "UNIQUE",  # UNIQUE
            "CLUSTERED",
            "AUTO_INCREMENT",  # AUTO_INCREMENT (MySQL)
            "UNSIGNED",  # UNSIGNED (MySQL)
            Sequence(  # REFERENCES reftable [ ( refcolumn) ]
                "REFERENCES",
                Ref("ColumnReferenceSegment"),
                # Foreign columns making up FOREIGN KEY constraint
                Ref("BracketedColumnReferenceListGrammar", optional=True),
            ),
            Ref("CommentClauseSegment"),
        ),
    )


@tsql_dialect.segment(replace=True)
class ColumnDefinitionSegment(BaseSegment):
    """A column definition, e.g. for CREATE TABLE or ALTER TABLE."""

    type = "column_definition"
    match_grammar = Sequence(
        Ref("StartSquareBracketSegment", optional=True),
        Ref("SingleIdentifierGrammar"),  # Column name
        Ref("EndSquareBracketSegment", optional=True),
        Ref("DatatypeSegment"),  # Column type
        Bracketed(Anything(), optional=True),  # For types like VARCHAR(100)
        AnyNumberOf(
            Ref("ColumnOptionSegment", optional=True),
        ),
    )


@tsql_dialect.segment(replace=True)
class CreateFunctionStatementSegment(BaseSegment):
    """A `CREATE FUNCTION` statement.

    This version in the TSQL dialect should be a "common subset" of the
    structure of the code for those dialects.
    postgres: https://www.postgresql.org/docs/9.1/sql-createfunction.html
    snowflake: https://docs.snowflake.com/en/sql-reference/sql/create-function.html
    bigquery: https://cloud.google.com/bigquery/docs/reference/standard-sql/user-defined-functions
    tsql/mssql : https://docs.microsoft.com/en-us/sql/t-sql/statements/create-function-transact-sql?view=sql-server-ver15
    """

    type = "create_function_statement"

    match_grammar = Sequence(
        "CREATE",
        Sequence("OR", "ALTER", optional=True),
        "FUNCTION",
        Anything(),
    )
    parse_grammar = Sequence(
        "CREATE",
        Sequence("OR", "ALTER", optional=True),
        "FUNCTION",
        Ref("SchemaNameSegment"),
        Ref("ObjectNameSegment"),
        Ref("FunctionParameterListGrammar"),
        Sequence(  # Optional function return type
            "RETURNS",
            Ref("DatatypeSegment"),
            optional=True,
        ),
        "AS",
        Ref("FunctionDefinitionGrammar"),
        Ref("GoStatementSegment", optional=True),
    )


@tsql_dialect.segment(replace=True)
class FunctionDefinitionGrammar(BaseSegment):
    """This is the body of a `CREATE FUNCTION AS` statement."""

    type = "function_statement"
    name = "function_statement"

    match_grammar = Sequence(Anything())


@tsql_dialect.segment()
class CreateProcedureStatementSegment(BaseSegment):
    """A `CREATE PROCEDURE` statement.
    https://docs.microsoft.com/en-us/sql/t-sql/statements/create-procedure-transact-sql?view=sql-server-ver15
    """

    type = "create_procedure_statement"

    match_grammar = Sequence(
        "CREATE",
        Sequence("OR", "ALTER", optional=True),
        OneOf("PROCEDURE", "PROC"),
        Ref("SchemaNameSegment", optional=True),
        Ref("ObjectNameSegment"),
        Ref("FunctionParameterListGrammar", optional=True),
        "AS",
        Ref("ProcedureDefinitionGrammar"),
        Ref("GoStatementSegment", optional=True),
    )

@tsql_dialect.segment()
class IfExpressionStatement(BaseSegment):
    """IF-ELSE statement.

    https://docs.microsoft.com/en-us/sql/t-sql/language-elements/if-else-transact-sql?view=sql-server-ver15
    """

    type = "if_else_statement"

    match_grammar = AnyNumberOf(
        Sequence(
            "IF",
            Ref("ExpressionSegment"),
            Ref("StatementSegment"),
        ),
        Sequence("ELSE", Ref("StatementSegment"), optional=True),
    )


@tsql_dialect.segment(replace=True)
class AlterTableStatementSegment(BaseSegment):
    """An `ALTER TABLE` statement."""

    type = "alter_table_statement"
    # https://docs.microsoft.com/en-us/sql/t-sql/statements/alter-table-transact-sql?view=sql-server-ver15
    # TODO: Rewrite to conform to TSQL
    match_grammar = Sequence(
        "ALTER",
        "TABLE",
        Ref("TableReferenceSegment"),
        Delimited(
            OneOf(
                # Table options
                Sequence(
                    Ref("ParameterNameSegment"),
                    Ref("EqualsSegment", optional=True),
                    OneOf(Ref("LiteralGrammar"), Ref("NakedIdentifierSegment")),
                ),
                # Add things
                Sequence(
                    OneOf("ADD", "ALTER"),
                    Ref.keyword("COLUMN", optional=True),
                    Ref("ColumnDefinitionSegment"),
                    OneOf(
                        Sequence(
                            OneOf("FIRST", "AFTER"), Ref("ColumnReferenceSegment")
                        ),
                        # Bracketed Version of the same
                        Ref("BracketedColumnReferenceListGrammar"),
                        optional=True,
                    ),
                ),
            ),
        ),
    )


@tsql_dialect.segment()
class AlterTableSwitchStatementSegment(BaseSegment):
    """An `ALTER TABLE SWITCH` statement."""

    type = "alter_table_statement"
    # https://docs.microsoft.com/en-us/sql/t-sql/statements/alter-table-transact-sql?view=sql-server-ver15
    # T-SQL's ALTER TABLE SWITCH grammar is different enough to core ALTER TABLE grammar to merit its own definition
    match_grammar = Sequence(
        "ALTER",
        "TABLE",
        Ref("SchemaNameSegment", optional=True),
        Ref("ObjectNameSegment"),
        "SWITCH",
        Sequence("PARTITION", Ref("NumericLiteralSegment"), optional=True),
        "TO",
        Ref("SchemaNameSegment", optional=True),
        Ref("ObjectNameSegment"),
        Sequence(
        "WITH",
        #"(",
        "TRUNCATE_TARGET",
        #Ref("EqualsSegment"),
        #"ON",")", 
        optional=True),
        Anything(),
    )

@tsql_dialect.segment(replace=True)
class ValuesClauseSegment(BaseSegment):
    """A `VALUES` clause like in `INSERT`."""

    type = "values_clause"
    match_grammar = Sequence(
        "VALUES",
        Delimited(
            Bracketed(
                Delimited(
                    Ref("LiteralGrammar"),
                    Ref("IntervalExpressionSegment"),
                    Ref("FunctionSegment"),
                    "DEFAULT",  # not in `FROM` clause, rule?
                    ephemeral_name="ValuesClauseElements",
                )
            ),
        ),
        Ref("AliasExpressionSegment", optional=True),
    )

@tsql_dialect.segment(replace=True)
class TransactionStatementSegment(BaseSegment):
    """A `COMMIT`, `ROLLBACK` or `TRANSACTION` statement."""

    type = "transaction_statement"
    match_grammar = Sequence(
        # ROLLBACK [ WORK ] [ AND [ NO ] CHAIN ]
        # BEGIN | END TRANSACTION | WORK
        # NOTE: "TO SAVEPOINT" is not yet supported
        # https://docs.snowflake.com/en/sql-reference/sql/begin.html
        # https://www.postgresql.org/docs/current/sql-end.html
        OneOf("BEGIN", "COMMIT", "ROLLBACK", "END"),
        OneOf("TRANSACTION", optional=True),
        Sequence("WITH","MARK", Ref("SingleIdentifierGrammar"), optional=True),
    )


@tsql_dialect.segment(replace=True)
class ObjectReferenceSegment(BaseSegment):
    """A reference to an object."""

    type = "object_reference"
    # match grammar (don't allow whitespace)
    match_grammar: Matchable = Delimited(
        Ref("SingleIdentifierGrammar"),
        delimiter=OneOf(
            Ref("DotSegment"), Sequence(Ref("DotSegment"), Ref("DotSegment"))
        ),
        terminator=OneOf(
            "ON",
            "AS",
            Ref("CommaSegment"),
            Ref("CastOperatorSegment"),
            Ref("StartSquareBracketSegment"),
            Ref("StartBracketSegment"),
            Ref("BinaryOperatorGrammar"),
            Ref("ColonSegment"),
            Ref("DelimiterSegment"),
            BracketedSegment,
        ),
        allow_gaps=False,
    )

    class ObjectReferencePart(NamedTuple):
        """Details about a table alias."""

        part: str  # Name of the part
        # Segment(s) comprising the part. Usuaully just one segment, but could
        # be multiple in dialects (e.g. BigQuery) that support unusual
        # characters in names (e.g. "-")
        segments: List[BaseSegment]

    @classmethod
    def _iter_reference_parts(cls, elem) -> Generator[ObjectReferencePart, None, None]:
        """Extract the elements of a reference and yield."""
        # trim on quotes and split out any dots.
        for part in elem.raw_trimmed().split("."):
            yield cls.ObjectReferencePart(part, [elem])

    def iter_raw_references(self) -> Generator[ObjectReferencePart, None, None]:
        """Generate a list of reference strings and elements.

        Each reference is an ObjectReferencePart. If some are split, then a
        segment may appear twice, but the substring will only appear once.
        """
        # Extract the references from those identifiers (because some may be quoted)
        for elem in self.recursive_crawl("identifier"):
            yield from self._iter_reference_parts(elem)

    def is_qualified(self):
        """Return if there is more than one element to the reference."""
        return len(list(self.iter_raw_references())) > 1

    def qualification(self):
        """Return the qualification type of this reference."""
        return "qualified" if self.is_qualified() else "unqualified"

    class ObjectReferenceLevel(Enum):
        """Labels for the "levels" of a reference.

        Note: Since SQLFluff does not have access to database catalog
        information, interpreting references will often be ambiguous.
        Typical example: The first part *may* refer to a schema, but that is
        almost always optional if referring to an object in some default or
        currently "active" schema. For this reason, use of this enum is optional
        and intended mainly to clarify the intent of the code -- no guarantees!
        Additionally, the terminology may vary by dialect, e.g. in BigQuery,
        "project" would be a more accurate term than "schema".
        """

        OBJECT = 1
        TABLE = 2
        SCHEMA = 3

    def extract_possible_references(
        self, level: Union[ObjectReferenceLevel, int]
    ) -> List[ObjectReferencePart]:
        """Extract possible references of a given level.

        "level" may be (but is not required to be) a value from the
        ObjectReferenceLevel enum defined above.

        NOTE: The base implementation here returns at most one part, but
        dialects such as BigQuery that support nesting (e.g. STRUCT) may return
        multiple reference parts.
        """
        level = self._level_to_int(level)
        refs = list(self.iter_raw_references())
        if len(refs) >= level:
            return [refs[-level]]
        return []

    @staticmethod
    def _level_to_int(level: Union[ObjectReferenceLevel, int]) -> int:
        # If it's an ObjectReferenceLevel, get the value. Otherwise, assume it's
        # an int.
        level = getattr(level, "value", level)
        assert isinstance(level, int)
        return level


# Need to include this to reset ObjectReferenceSegment reference
@tsql_dialect.segment(replace=True)
class TableReferenceSegment(ObjectReferenceSegment):
    """A reference to an table, CTE, subquery or alias."""

    type = "table_reference"


# Need to include this to reset ObjectReferenceSegment reference
@tsql_dialect.segment(replace=True)
class SchemaReferenceSegment(ObjectReferenceSegment):
    """A reference to a schema."""

    type = "schema_reference"


# Need to include this to reset ObjectReferenceSegment reference
@tsql_dialect.segment(replace=True)
class DatabaseReferenceSegment(ObjectReferenceSegment):
    """A reference to a database."""

    type = "database_reference"


# Need to include this to reset ObjectReferenceSegment reference
@tsql_dialect.segment(replace=True)
class IndexReferenceSegment(ObjectReferenceSegment):
    """A reference to an index."""

    type = "index_reference"


# Need to include this to reset ObjectReferenceSegment reference
@tsql_dialect.segment(replace=True)
class ExtensionReferenceSegment(ObjectReferenceSegment):
    """A reference to an extension."""

    type = "extension_reference"


# Need to include this to reset ObjectReferenceSegment reference
@tsql_dialect.segment(replace=True)
class ColumnReferenceSegment(ObjectReferenceSegment):
    """A reference to column, field or alias."""

    type = "column_reference"


# Need to include this to reset ObjectReferenceSegment reference
@tsql_dialect.segment(replace=True)
class WildcardIdentifierSegment(ObjectReferenceSegment):
    """Any identifier of the form a.b.*.

    This inherits iter_raw_references from the
    ObjectReferenceSegment.
    """

    type = "wildcard_identifier"
    match_grammar = Sequence(
        # *, blah.*, blah.blah.*, etc.
        AnyNumberOf(
            Sequence(Ref("SingleIdentifierGrammar"), Ref("DotSegment"), allow_gaps=True)
        ),
        Ref("StarSegment"),
        allow_gaps=False,
    )

    def iter_raw_references(self):
        """Generate a list of reference strings and elements.

        Each element is a tuple of (str, segment). If some are
        split, then a segment may appear twice, but the substring
        will only appear once.
        """
        # Extract the references from those identifiers (because some may be quoted)
        for elem in self.recursive_crawl("identifier", "star"):
            yield from self._iter_reference_parts(elem)


@tsql_dialect.segment(replace=True)
class JoinClauseSegment(BaseSegment):
    """Any number of join clauses, including the `JOIN` keyword."""

    type = "join_clause"
    match_grammar = Sequence(
        # NB These qualifiers are optional
        # TODO: Allow nested joins like:
        # ....FROM S1.T1 t1 LEFT JOIN ( S2.T2 t2 JOIN S3.T3 t3 ON t2.col1=t3.col1) ON tab1.col1 = tab2.col1
        OneOf(
            "CROSS",
            "INNER",
            Sequence(
                OneOf(
                    "FULL",
                    "LEFT",
                    "RIGHT",
                ),
                Ref.keyword("OUTER", optional=True),
            ),
            optional=True,
        ),
        "JOIN",
        Indent,
        Sequence(
            Ref("FromExpressionElementSegment"),
            Conditional(Dedent, indented_using_on=False),
            # ON clause
            Ref("JoinOnConditionSegment", optional=True), #optional for CROSS JOIN
            Conditional(Indent, indented_using_on=False),
        ),
        Dedent,
    )

    def get_eventual_alias(self) -> AliasInfo:
        """Return the eventual table name referred to by this join clause."""
        from_expression_element = self.get_child("from_expression_element")
        return from_expression_element.get_eventual_alias()

@tsql_dialect.segment()
class ProcedureDefinitionGrammar(BaseSegment):
    """This is the body of a `CREATE OR ALTER PROCEDURE AS` statement."""

    type = "procedure_statement"
    name = "procedure_statement"

    match_grammar = Sequence(Anything())


@tsql_dialect.segment(replace=True)
class IntervalExpressionSegment(BaseSegment):
    """An interval expression segment."""

    type = "interval_expression"
    match_grammar = Nothing()


@tsql_dialect.segment(replace=True)
class CreateExtensionStatementSegment(BaseSegment):
    """A `CREATE EXTENSION` statement.

    https://www.postgresql.org/docs/9.1/sql-createextension.html
    """

    type = "create_extension_statement"
    match_grammar = Nothing()


@tsql_dialect.segment(replace=True)
class CreateModelStatementSegment(BaseSegment):
    """A BigQuery `CREATE MODEL` statement."""

    type = "create_model_statement"
    match_grammar = Nothing()


@ansi_dialect.segment(replace=True)
class DropModelStatementSegment(BaseSegment):
    """A `DROP MODEL` statement."""

    type = "drop_MODELstatement"
    match_grammar = Nothing()


@tsql_dialect.segment(replace=True)
class LimitClauseSegment(BaseSegment):
    """A `LIMIT` clause like in `SELECT`."""

    type = "limit_clause"
    match_grammar = Nothing()

@tsql_dialect.segment(replace=True)
class OverlapsClauseSegment(BaseSegment):
    """An `OVERLAPS` clause like in `SELECT."""

    type = "overlaps_clause"
    match_grammar = Nothing()

@tsql_dialect.segment(replace=True)
class NamedWindowSegment(BaseSegment):
    """A WINDOW clause."""

    type = "named_window"
    match_grammar = Nothing()

@tsql_dialect.segment(replace=True)
class CreateViewStatementSegment(BaseSegment):
    """A `CREATE VIEW` statement."""

    type = "create_view_statement"
    # https://docs.microsoft.com/en-us/sql/t-sql/statements/create-view-transact-sql?view=sql-server-ver15#examples
    match_grammar = Sequence(
        "CREATE",
        Sequence("OR", "ALTER", optional=True),
        "VIEW",
        Ref("SchemaNameSegment", optional=True),
        Ref("ObjectNameSegment"),
        "AS",
        Ref("SelectableGrammar"),
        Ref("GoStatementSegment", optional=True),
    )


@tsql_dialect.segment(replace=True)
class SelectClauseSegment(BaseSegment):
    """A group of elements in a select target statement."""

    type = "select_clause"
    match_grammar = StartsWith(
        Sequence("SELECT", Ref("WildcardExpressionSegment", optional=True)),
        terminator=OneOf(
            "FROM",
            "WHERE",
            "ORDER",
            Ref("SetOperatorSegment"),
        ),
        enforce_whitespace_preceeding_terminator=True,
    )

    parse_grammar = Sequence(
        "SELECT",
        Ref("SelectClauseModifierSegment", optional=True),
        Indent,
        Delimited(
            Ref("SelectClauseElementSegment"),
            allow_trailing=True,
        ),
        # NB: The Dedent for the indent above lives in the
        # SelectStatementSegment so that it sits in the right
        # place corresponding to the whitespace.
    )


@tsql_dialect.segment(replace=True)
class SelectClauseElementSegment(BaseSegment):
    """An element in the targets of a select statement."""

    type = "select_clause_element"
    # Important to split elements before parsing, otherwise debugging is really hard.
    match_grammar = GreedyUntil(
        "FROM",
        "WHERE",
        "ORDER",
        Ref("CommaSegment"),
        Ref("SetOperatorSegment"),
        enforce_whitespace_preceeding_terminator=True,
    )

    parse_grammar = OneOf(
        # *, blah.*, blah.blah.*, etc.
        Ref("WildcardExpressionSegment"),
        Sequence(
            Ref("BaseExpressionElementGrammar"),
            Ref("AliasExpressionSegment", optional=True),
        ),
    )


@tsql_dialect.segment(replace=True)
class OrderByClauseSegment(BaseSegment):
    """A `ORDER BY` clause like in `SELECT`."""

    type = "orderby_clause"
    match_grammar = StartsWith(
        "ORDER",
        terminator=OneOf(
            "HAVING",
            # For window functions
            "ROWS",
        ),
    )
    parse_grammar = Sequence(
        "ORDER",
        "BY",
        Indent,
        Delimited(
            Sequence(
                OneOf(
                    Ref("ColumnReferenceSegment"),
                    # Can `ORDER BY 1`
                    Ref("NumericLiteralSegment"),
                    # Can order by an expression
                    Ref("ExpressionSegment"),
                ),
                OneOf("ASC", "DESC", optional=True),
                # NB: This isn't really ANSI, and isn't supported in Mysql, but
                # is supported in enough other dialects for it to make sense here
                # for now.
                Sequence("NULLS", OneOf("FIRST", "LAST"), optional=True),
            ),
        ),
        Dedent,
    )


@tsql_dialect.segment(replace=True)
class GroupByClauseSegment(BaseSegment):
    """A `GROUP BY` clause like in `SELECT`."""

    type = "groupby_clause"
    match_grammar = StartsWith(
        Sequence("GROUP", "BY"),
        terminator=OneOf("ORDER", "HAVING"),
        enforce_whitespace_preceeding_terminator=True,
    )
    parse_grammar = Sequence(
        "GROUP",
        "BY",
        Indent,
        Delimited(
            OneOf(
                Ref("ColumnReferenceSegment"),
                # Can `GROUP BY 1`
                Ref("NumericLiteralSegment"),
                # Can `GROUP BY coalesce(col, 1)`
                Ref("ExpressionSegment"),
            ),
            terminator=OneOf("ORDER", "HAVING"),
        ),
        Dedent,
    )


@tsql_dialect.segment(replace=True)
class HavingClauseSegment(BaseSegment):
    """A `HAVING` clause like in `SELECT`."""

    type = "having_clause"
    match_grammar = StartsWith(
        "HAVING",
        terminator=OneOf("ORDER"),
        enforce_whitespace_preceeding_terminator=True,
    )
    parse_grammar = Sequence(
        "HAVING",
        Indent,
        OptionallyBracketed(Ref("ExpressionSegment")),
        Dedent,
    )


@tsql_dialect.segment(replace=True)
class SetOperatorSegment(BaseSegment):
    """A set operator such as Union, Minus, Except or Intersect."""

    type = "set_operator"
    match_grammar = OneOf(
        Sequence("UNION", OneOf("DISTINCT", "ALL", optional=True)),
        "INTERSECT",
        "EXCEPT",
        exclude=Sequence("EXCEPT", Bracketed(Anything())),
    )


@tsql_dialect.segment(replace=True)
class MLTableExpressionSegment(BaseSegment):
    """An ML table expression."""
    # https://docs.microsoft.com/en-us/sql/t-sql/queries/predict-transact-sql?view=sql-server-ver15
    type = "ml_table_expression"
    # E.g. ML.WEIGHTS(MODEL `project.dataset.model`)
    match_grammar = Sequence(
        "PREDICT",
        Bracketed(
            Sequence("MODEL", Ref("EqualsSegment"), Ref("ObjectReferenceSegment")),
            Sequence("DATA", Ref("EqualsSegment"), Ref("ObjectReferenceSegment")),
        ),
        Anything() # TODO
    )


@tsql_dialect.segment(replace=True)
class StatementSegment(ansi_dialect.get_segment("StatementSegment")):  # type: ignore
    """Overriding StatementSegment to allow for additional segment parsing."""

    parse_grammar = ansi_dialect.get_segment("StatementSegment").parse_grammar.copy(
        insert=[
            Ref("CreateProcedureStatementSegment"),
            Ref("IfExpressionStatement"),
            Ref("AlterTableSwitchStatementSegment"),
            Ref("JoinClauseSegment"),
            Ref("ObjectReferenceSegment"),
        ],
        remove=[
            Ref("DescribeStatementSegment"),
            Ref("ExplainStatementSegment") # to be added back into tsql_PDW dialect
        ]
    )

