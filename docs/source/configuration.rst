.. _config:

Configuration
=============

SQLFluff accepts configuration either through the command line or
through configuration files. There is *rough* parity between the
two approaches with the exception that *templating* configuration
must be done via a file, because it otherwise gets slightly complicated.

For details of what's available on the command line check out
the :ref:`cliref`.

For file based configuration *SQLFluff* will look for the following
files in order. Later files will (if found) will be used to overwrite
any vales read from earlier files.

- :code:`setup.cfg`
- :code:`tox.ini`
- :code:`pep8.ini`
- :code:`.sqlfluff`
- :code:`pyproject.toml`

Within these files, the first four will be read like an `cfg file`_, and
*SQLFluff* will look for sections which start with *SQLFluff*, and where
subsections are delimited by a semicolon. For example the *jinjacontext*
section will be indicated in the section started with
*[sqlfluff:jinjacontext]*.

For the `pyproject.toml file`_, all valid sections start with `tool.sqlfluff`
and subsections are delimited by a dot. For example the *jinjacontext* section
will be indicated in the section started with *[tool.sqlfluff.jinjacontext]*.

For example

.. code-block:: toml

    [tool.sqlfluff.core]
    templater = "jinja"
    sql_file_exts = ".sql,.sql.j2,.dml,.ddl"

    [tool.sqlfluff.indentation]
    indented_joins = false
    indented_using_on = true
    template_blocks_indent = false

    [tool.sqlfluff.templater]
    unwrap_wrapped_queries = true

    [tool.sqlfluff.templater.jinja]
    apply_dbt_builtins = true

.. _`cfg file`: https://docs.python.org/3/library/configparser.html
.. _`pyproject.toml file`: https://www.python.org/dev/peps/pep-0518/

Nesting
-------

**SQLFluff** uses **nesting** in its configuration files, with files
closer *overriding* (or *patching*, if you will) values from other files.
That means you'll end up with a final config which will be a patchwork
of all the values from the config files loaded up to that path. The exception
to this is the value for `templater`, which cannot be set in config files in
subdirectories of the working directory.
You don't **need** any config files to be present to make *SQLFluff*
work. If you do want to override any values though SQLFluff will use
files in the following locations in order, with values from later
steps overriding those from earlier:

0. *[...and this one doesn't really count]* There's a default config as
   part of the SQLFluff package. You can find this below, in the
   :ref:`defaultconfig` section.
1. It will look in the user's os-specific app config directory. On OSX this is
   `~/Library/Preferences/sqlfluff`, Unix is `~/.config/sqlfluff`, Windows is
   `<home>\\AppData\\Local\\sqlfluff\\sqlfluff`, for any of the filenames
   above in the main :ref:`config` section. If multiple are present, they will
   *patch*/*override* each other in the order above.
2. It will look for the same files in the user's home directory (~).
3. It will look for the same files in the current working directory.
4. *[if parsing a file in a subdirectory of the current working directory]*
   It will look for the same files in every subdirectory between the
   current working dir and the file directory.
5. It will look for the same files in the directory containing the file
   being linted.

This whole structure leads to efficient configuration, in particular
in projects which utilise a lot of complicated templating.

.. _templateconfig:

Jinja Templating Configuration
------------------------------

When thinking about Jinja templating there are two different kinds of things
that a user might want to fill into a templated file, *variables* and
*functions/macros*. Currently *functions* aren't implemented in any
of the templaters.

Variable Templating
^^^^^^^^^^^^^^^^^^^

Variables are available in the *jinja* and *python* templaters. By default
the templating engine will expect variables for templating to be available
in the config, and the templater will be look in the section corresponding
to the context for that templater. By convention, the config for the *jinja*
templater is found in the *sqlfluff:templater:jinja:context* section and the
config for the *python* templater is found in the
*sqlfluff:templater:python:context* section.

For example, if passed the following *.sql* file:

.. code-block:: jinja

    SELECT {{ num_things }} FROM {{ tbl_name }} WHERE id > 10 LIMIT 5

...and the following configuration in *.sqlfluff* in the same directory:

.. code-block:: cfg

    [sqlfluff:templater:jinja:context]
    num_things=456
    tbl_name=my_table

...then before parsing, the sql will be transformed to:

.. code-block:: sql

    SELECT 456 FROM my_table WHERE id > 10 LIMIT 5

.. note::

    If there are variables in the template which cannot be found in
    the current configuration context, then this will raise a `SQLTemplatingError`
    and this will appear as a violation without a line number, quoting
    the name of the variable that couldn't be found.

Complex Variable Templating
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two more advanced features of variable templating are *case sensitivity*
and *native python types*. Both are illustrated in the following example:

.. code-block:: cfg

    [sqlfluff:templater:jinja:context]
    my_list=['a', 'b', 'c']
    MY_LIST=("d", "e", "f")
    my_where_dict={"field_1": 1, "field_2": 2}

.. code-block:: jinja

    SELECT
        {% for elem in MY_LIST %}
            '{{elem}}' {% if not loop.last %}||{% endif %}
        {% endfor %} as concatenated_list
    FROM tbl
    WHERE
        {% for field, value in my_where_dict.items() %}
            {{field}} = {{value}} {% if not loop.last %}and{% endif %}
        {% endfor %}

...will render as...

.. code-block:: sql

    SELECT
        'd' || 'e' || 'f' as concatenated_list
    FROM tbl
    WHERE
        field_1 = 1 and field_2 = 2

Note that the variable was replaced in a case sensitive way and that the
settings in the config file were interpreted as native python types.

Macro Templating
^^^^^^^^^^^^^^^^

Macros (which also look and feel like *functions* are available only in the
*jinja* templater. Similar to `Variable Templating`_, these are specified in
config files, what's different in this case is how they are named. Similar to
the *context* section above, macros are configured separately in the *macros*
section of the config. Consider the following example.

If passed the following *.sql* file:

.. code-block:: jinja

    SELECT {{ my_macro(6) }} FROM some_table

...and the following configuration in *.sqlfluff* in the same directory (note
the tight control of whitespace):

.. code-block:: cfg

    [sqlfluff:templater:jinja:macros]
    a_macro_def = {% macro my_macro(something) %}{{something}} + {{something * 2}}{% endmacro %}

...then before parsing, the sql will be transformed to:

.. code-block:: sql

    SELECT 6 + 12 FROM some_table

Note that in the code block above, the variable name in the config is
*a_macro_def*, and this isn't apparently otherwise used anywhere else.
Broadly this is accurate, however within the configuration loader this will
still be used to overwrite previous *values* in other config files. As such
this introduces the idea of config *blocks* which could be selectively
overwritten by other configuration files downstream as required.

In addition to macros specified in the config file, macros can also be
loaded from a file or folder. The path to this macros folder must be
specified in the config file to function as below:

.. code-block:: cfg

    [sqlfluff:templater:jinja]
    load_macros_from_path=my_macros

In this case, SQLFluff will load macros from any :code:`.sql` file found at the
path specified on this variable. The path is interpreted *relative to the
config file*, and therefore if the config file above was found at
:code:`/home/my_project/.sqlfluff` then SQLFluff will look for macros in the
folder :code:`/home/my_project/my_macros/`. Alternatively the path can also
be a :code:`.sql` itself. Any macros defined in the config will always take
precedence over a macro defined in the path.


.. note::

    Throughout the templating process **whitespace** will still be treated
    rigorously, and this includes **newlines**. In particular you may choose
    to provide your *dummy* macros in your configuration with different to
    the actual macros you may be using in production.

    **REMEMBER:** The purpose of providing the option of macros is to *enable*
    the parsing of templated sql without it being a blocker. It shouldn't
    be a requirement that the *templating* is accurate - only so far as that
    is required to enable the *parsing* and *linting* to be helpful.

Builtin Macro Blocks
^^^^^^^^^^^^^^^^^^^^

One of the main use cases which inspired *SQLFluff* as a project was `dbt`_.
It uses jinja templating extensively and leads to some users maintaining large
repositories of sql files which could potentially benefit from some linting.

.. note::
    *SQLFluff* has now a tighter integration with dbt through the "dbt" templater.
    It is the recommended templater for dbt projects and removes the need for the
    overwrites described in this section.

    To use the dbt templater, go to `Dbt Project Configuration`_.

*SQLFluff* anticipates this use case and provides some built in macro blocks
in the `Default Configuration`_ which assist in getting started with `dbt`_
projects. In particular it provides mock objects for:

* *ref*: The mock version of this provided simply returns the model reference
  as the name of the table. In most cases this is sufficient.
* *config*: A regularly used macro in `dbt`_ to set configuration values. For
  linting purposes, this makes no difference and so the provided macro simply
  returns nothing.

.. note::
    If there are other builtin macros which would make your life easier,
    consider submitting the idea (or even better a pull request) on `github`_.

.. _`dbt`: https://www.getdbt.com/
.. _`github`: https://www.github.com/sqlfluff/sqlfluff

.. _dbt-project-configuration:

Library Templating
^^^^^^^^^^^^^^^^^^

If using *SQLFluff* for dbt with jinja as your templater, you may have library
function calls within your sql files that can not be templated via the
normal macro templating mechanisms:

.. code-block:: jinja

    SELECT foo, bar FROM baz {{ dbt_utils.group_by(2) }}

To template these libraries, you can use the `sqlfluff:jinja:library_path`
config option:

.. code-block:: cfg

    [sqlfluff:templater:jinja]
    library_path=sqlfluff_libs

This will pull in any python modules from that directory and allow sqlfluff
to use them for templated. In the above example, you might define a file at
`sqlfluff_libs/dbt_utils.py` as:

.. code-block:: python

    def group_by(n):
        return "GROUP BY 1,2"


dbt Project Configuration
-------------------------

.. note::
    dbt templating is a new feature added in 0.4.0 and has not benefited
    from widespread use and testing yet! If you encounter an issue, please
    let us know in a GitHub issue or on the SQLFluff slack workspace.

dbt is not the default templater for *SQLFluff* (it is Jinja). For using
*SQLFluff* with a dbt project, users can either use the `jinja` templater
(which may be slightly faster, but will not support the full spectrum of
macros) or the `dbt` templater, which uses dbt itself to render the
sql (meaning that there is a much more reliable representation of macros,
but a potential performance hit accordingly). At this stage we recommend
that users try both approaches and choose according to the method that
they intend to use *SQLFluff*.

A simple rule of thumb might be:

- If you are using *SQLFluff* in a CI/CD context, where speed is not
  critical but accuracy in rendering sql is, then the `dbt` templater
  may be more appropriate.
- If you are using *SQLFluff* in an IDE or on a git hook, where speed
  of response may be more important, then the `jinja` templater may
  be more appropriate.

In order to get started using *SQLFluff* with a dbt project you will
need the following configuration:

In *.sqlfluff*:

.. code-block:: cfg

    [sqlfluff]
    templater = dbt

In *.sqlfluffignore*:

.. code-block::

    target/
    dbt_modules/
    macros/

You can set the dbt project directory, profiles directory and profile with:

.. code-block::

    [sqlfluff:templater:dbt]
    project_dir = <relative or absolute path to dbt_project directory>
    profiles_dir = <relative or absolute path to the directory that contains the profiles.yml file>
    profile = <dbt profile>

Known Caveats
^^^^^^^^^^^^^

- To use the dbt templater, you must set `templater = dbt` in the `.sqlfluff`
  config file in the directory where sqlfluff is run. The templater cannot
  be changed in `.sqlfluff` files in subdirectories.
- In SQLFluff 0.4.0 using the dbt templater requires that all files
  within the root and child directories of the dbt project must be part
  of the project. If there are deployment scripts which refer to SQL files
  not part of the project for instance, this will result in an error.
  You can overcome this by adding any non-dbt project SQL files to
  .sqlfluffignore.


CLI Arguments
-------------

You already know you can pass arguments (:code:`--verbose`,
:code:`--exclude-rules`, etc.) through the CLI commands (:code:`lint`,
:code:`fix`, etc.):

.. code-block:: console

    $ sqlfluff lint my_code.sql -v --exclude-rules L022,L027

You might have arguments that you pass through every time, e.g rules you
*always* want to ignore. These can also be configured:

.. code-block:: cfg

    [sqlfluff]
    verbose = 1
    exclude_rules = L022,L027

Note that while the :code:`exclude_rules` config looks similar to the
above example, the :code:`verbose` config has an integer value. This is
because :code:`verbose` is *stackable* meaning there are multiple levels
of verbosity that are available for configuration. See :ref:`cliref` for
more details about the available CLI arguments.

Ignoring Errors & Files
-----------------------

Ignoring individual lines
^^^^^^^^^^^^^^^^^^^^^^^^^

Similar to `flake8's ignore`_, individual lines can be ignored by adding
:code:`-- noqa` to the end of the line. Additionally, specific rules can
be ignored by quoting their code or the category.

.. code-block:: sql

    -- Ignore all errors
    SeLeCt  1 from tBl ;    -- noqa

    -- Ignore rule L014 & rule L030
    SeLeCt  1 from tBl ;    -- noqa: L014,L030

    -- Ignore all parsing errors
    SeLeCt from tBl ;       -- noqa: PRS

.. _`flake8's ignore`: https://flake8.pycqa.org/en/3.1.1/user/ignoring-errors.html#in-line-ignoring-errors

Ignoring line ranges
^^^^^^^^^^^^^^^^^^^^

Similar to `pylint's "pylint" directive"`_, ranges of lines can be ignored by
adding :code:`-- noqa:disable=<rule>[,...] | all` to the line. Following this
directive, specified rules (or all rules, if "all" was specified) will be
ignored until a corresponding `-- noqa:enable=<rule>[,...] | all` directive.

.. code-block:: sql

    -- Ignore rule L012 from this line forward
    SELECT col_a a FROM foo --noqa: disable=L012

    -- Ignore all rules from this line forward
    SELECT col_a a FROM foo --noqa: disable=all

    -- Enforce all rules from this line forward
    SELECT col_a a FROM foo --noqa: enable=all


.. _`pylint's "pylint" directive"`: http://pylint.pycqa.org/en/latest/user_guide/message-control.html

.. _sqlfluffignore:

.sqlfluffignore
^^^^^^^^^^^^^^^

Similar to `Git's`_ :code:`.gitignore` and `Docker's`_ :code:`.dockerignore`,
SQLFluff supports a :code:`.sqlfluffignore` file to control which files are and
aren't linted. Under the hood we use the python `pathspec library`_ which also
has a brief tutorial in their documentation.

An example of a potential :code:`.sqlfluffignore` placed in the root of your
project would be:

.. code-block:: cfg

    # Comments start with a hash.

    # Ignore anything in the "temp" path
    /path/

    # Ignore anything called "testing.sql"
    testing.sql

    # Ignore any ".tsql" files
    *.tsql

Ignore files can also be placed in subdirectories of a path which is being
linted and the sub files will also be applied within that subdirectory.


.. _`Git's`: https://git-scm.com/docs/gitignore#_pattern_format
.. _`Docker's`: https://docs.docker.com/engine/reference/builder/#dockerignore-file
.. _`pathspec library`: https://python-path-specification.readthedocs.io/

.. _defaultconfig:

Default Configuration
---------------------

The default configuration is as follows, note the `Builtin Macro Blocks`_ in
section *[sqlfluff:templater:jinja:macros]* as referred to above.

.. literalinclude:: ../../src/sqlfluff/core/default_config.cfg
   :language: cfg
   :linenos:
