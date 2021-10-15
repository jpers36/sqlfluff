"""A list of all SQL key words.

https://docs.microsoft.com/en-us/sql/t-sql/language-elements/reserved-keywords-transact-sql?view=sql-server-ver15
"""

RESERVED_KEYWORDS = [
    "ADD",
    "ALL",
    "ALTER",
    "AND",
    "ANSI_DEFAULTS",
    "ANSI_NULL_DFLT_OFF",
    "ANSI_NULL_DFLT_ON",
    "ANSI_NULLS",
    "ANSI_PADDING",
    "ANSI_WARNINGS",
    "ANY",
    "ARITHABORT",
    "ARITHIGNORE",
    "AS",
    "ASC",
    "AUTHORIZATION",
    "BACKUP",
    "BEGIN",
    "BETWEEN",
    "BREAK",
    "BROWSE",
    "BULK",
    "BY",
    "CASCADE",
    "CASE",
    "CHECK",
    "CHECKPOINT",
    "CLOSE",
    "CLUSTERED",
    "COALESCE",
    "COLLATE",
    "COLUMN",
    "COMMIT",
    "COMPUTE",
    "CONCAT_NULL_YIELDS_NULL",
    "CONSTRAINT",
    "CONTAINS",
    "CONTAINSTABLE",
    "CONTINUE",
    "CONVERT",
    "CREATE",
    "CROSS",
    "CURRENT_DATE",
    "CURRENT_TIME",
    "CURRENT_TIMESTAMP",
    "CURRENT_USER",
    "CURRENT",
    "CURSOR_CLOSE_ON_COMMIT",
    "CURSOR",
    "DATABASE",
    "DATEFIRST",
    "DATEFORMAT",
    "DBCC",
    "DEADLOCK_PRIORITY",
    "DEALLOCATE",
    "DECLARE",
    "DEFAULT",
    "DELETE",
    "DENY",
    "DESC",
    "DISK",
    "DISTINCT",
    "DISTRIBUTED",
    "DOUBLE",
    "DROP",
    "DUMP",
    "ELSE",
    "END",
    "ERRLVL",
    "ESCAPE",
    "EXCEPT",
    "EXEC",
    "EXECUTE",
    "EXISTS",
    "EXIT",
    "EXTERNAL",
    "FETCH",
    "FILE",
    "FILLFACTOR",
    "FIPS_FLAGGER",
    "FMTONLY",
    "FOR",
    "FORCEPLAN",
    "FOREIGN",
    "FREETEXT",
    "FREETEXTTABLE",
    "FROM",
    "FULL",
    "FUNCTION",
    "GOTO",
    "GRANT",
    "GROUP",
    "HAVING",
    "HOLDLOCK",
    "IDENTITY_INSERT",
    "IDENTITY_INSERT",
    "IDENTITY",
    "IDENTITYCOL",
    "IF",
    "IMPLICIT_TRANSACTIONS",
    "IN",
    "INDEX",
    "INNER",
    "INSERT",
    "INTERSECT",
    "INTO",
    "IS",
    "JOIN",
    "KEY",
    "KILL",
    "LANGUAGE",
    "LEFT",
    "LIKE",
    "LINENO",
    "LOAD",
    "LOCK_TIMEOUT",
    "MERGE",
    "NATIONAL",
    "NOCHECK",
    "NOCOUNT",
    "NOEXEC",
    "NONCLUSTERED",
    "NOT",
    "NULL",
    "NULLIF",
    "NUMERIC_ROUNDABORT",
    "OF",
    "OFF",
    "OFFSETS",
    "OFFSETS",
    "ON",
    "OPEN",
    "OPENDATASOURCE",
    "OPENQUERY",
    "OPENROWSET",
    "OPENXML",
    "OPTION",
    "OR",
    "ORDER",
    "OUTER",
    "OVER",
    "PARSEONLY",
    "PERCENT",
    "PIVOT",
    "PLAN",
    "PRECISION",
    "PRIMARY",
    "PRINT",
    "PROC",
    "PROCEDURE",
    "PUBLIC",
    "QUERY_GOVERNOR_COST_LIMIT",
    "QUOTED_IDENTIFIER",
    "RAISERROR",
    "READ",
    "READTEXT",
    "RECONFIGURE",
    "REFERENCES",
    "REMOTE_PROC_TRANSACTIONS",
    "REPLICATION",
    "RESTORE",
    "RESTRICT",
    "RESULT CACHING (Preview)",
    "RETURN",
    "REVERT",
    "REVOKE",
    "RIGHT",
    "ROLLBACK",
    "ROWCOUNT",
    "ROWCOUNT",
    "ROWGUIDCOL",
    "RULE",
    "SAVE",
    "SCHEMA",
    "SECURITYAUDIT",
    "SELECT",
    "SEMANTICKEYPHRASETABLE",
    "SEMANTICSIMILARITYDETAILSTABLE",
    "SEMANTICSIMILARITYTABLE",
    "SESSION_USER",
    "SET",
    "SETUSER",
    "SHOWPLAN_ALL",
    "SHOWPLAN_TEXT",
    "SHOWPLAN_XML",
    "SHUTDOWN",
    "SOME",
    "STATISTICS IO",
    "STATISTICS PROFILE",
    "STATISTICS TIME",
    "STATISTICS XML",
    "STATISTICS",
    "SYSTEM_USER",
    "TABLE",
    "TABLESAMPLE",
    "TEXTSIZE",
    "TEXTSIZE",
    "THEN",
    "TO",
    "TOP",
    "TRAN",
    "TRANSACTION ISOLATION LEVEL",
    "TRANSACTION",
    "TRIGGER",
    "TRUNCATE",
    "TRY_CONVERT",
    "TSEQUAL",
    "UNION",
    "UNIQUE",
    "UNPIVOT",
    "UPDATE",
    "UPDATETEXT",
    "USE",
    "USER",
    "VALUES",
    "VARYING",
    "VIEW",
    "WAITFOR",
    "WHEN",
    "WHERE",
    "WHILE",
    "WITH",
    "WRITETEXT",
    "XACT_ABORT",
]


UNRESERVED_KEYWORDS = [
    "CAST",
    "COLUMNSTORE",
    "CONCAT",
    "D",
    "DATEADD",
    "DATEDIFF",
    "DATEDIFF_BIG",
    "DATENAME",
    "DAY",
    "DAYOFYEAR",
    "DD",
    "DISTRIBUTION",  # Azure Synapse Analytics specific
    "DW",
    "DY",
    "EXPAND",
    "FAST",
    "HASH",  # Azure Synapse Analytics specific
    "HH",
    "HOUR",
    "KEEP",
    "KEEPFIXED",
    "LABEL", # Azure Synapse Analytics specific, reserved keyword but could break TSQL parsing to add there
    "M",
    "MAX_GRANT_PERCENT",
    "MIN_GRANT_PERCENT",
    "MAXDOP",
    "MAXRECURSION",
    "MCS",
    "MI",
    "MICROSECOND",
    "MILLISECOND",
    "MINUTE",
    "MM",
    "MONTH",
    "MS",
    "N",
    "NANOSECOND",
    "NS",
    "Q",
    "QQ",
    "QUARTER",
    "QUERYTRACEON",
    "REPLICATE",  # Azure Synapse Analytics specific
    "ROBUST",
    "ROUND_ROBIN",  # Azure Synapse Analytics specific
    "S",
    "SECOND",
    "SS",
    "STRING_AGG",
    "SWITCH",
    "TRUNCATE_TARGET",  # Azure Synapse Analytics specific
    "W",
    "WEEK",
    "WEEKDAY",
    "WITHIN",
    "WK",
    "WW",
    "YEAR",
    "Y",
    "YY",
    "YYYY",
]
