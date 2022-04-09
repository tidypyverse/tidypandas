from textwrap import dedent

from pandas.io.formats.format import (
    DataFrameFormatter, 
    format_array, 
    DataFrameRenderer, 
    save_to_buffer
)
from pandas.io.formats.html import HTMLFormatter
from pandas import MultiIndex

class TidyDataFrameFormatter(DataFrameFormatter):
    def format_col(self, i: int):
        """Format column, add dtype ahead"""
        frame = self.tr_frame
        formatter = self._get_formatter(i)
        dtype = frame.iloc[:, i].dtype.name

        return [f'<{dtype}>'] + format_array(
            frame.iloc[:, i]._values,
            formatter,
            float_format=self.float_format,
            na_rep=self.na_rep,
            space=self.col_space.get(frame.columns[i]),
            decimal=self.decimal,
            leading_space=self.index,
        )


    def get_strcols(self):
        """
        Render a DataFrame to a list of columns (as lists of strings).
        """
        strcols = self._get_strcols_without_index()

        if self.index:
            #           dtype
            str_index = [""] + self._get_formatted_index(self.tr_frame)
            strcols.insert(0, str_index)

        return strcols


class TidyHTMLFormatter(HTMLFormatter):
    def _write_body(self, indent: int): # -> None:
        self.write("<tbody>", indent)
        fmt_values = self._get_formatted_values()

        column_types = [v[0] for k, v in fmt_values.items()]
        self._write_row_column_types(column_types)

        fmt_values = {k: v[1:] for k, v in fmt_values.items()}
        
        # write values
        if self.fmt.index and isinstance(self.frame.index, MultiIndex):
            self._write_hierarchical_rows(fmt_values, indent + self.indent_delta)
        else:
            self._write_regular_rows(fmt_values, indent + self.indent_delta)

        self.write("</tbody>", indent)

    def _write_row_column_types(self, column_types):
        self.write_tr([""] + column_types
                      , header=False
                      , tags={i: 'style="color:#606060;font-style:oblique;"' for i in range(len(column_types)+1)}
                      )


class TidyNotebookFormatter(TidyHTMLFormatter):
    """
    Internal class for formatting output data in html for display in Jupyter
    Notebooks. This class is intended for functionality specific to
    DataFrame._repr_html_() and DataFrame.to_html(notebook=True)
    """

    def _get_formatted_values(self): #-> dict[int, list[str]]:
        return {i: self.fmt.format_col(i) for i in range(self.ncols)}

    def _get_columns_formatted_values(self): # -> list[str]:
        return self.columns.format()

    def write_style(self): # -> None:
        # We use the "scoped" attribute here so that the desired
        # style properties for the data frame are not then applied
        # throughout the entire notebook.
        template_first = """\
            <style scoped>"""
        template_last = """\
            </style>"""
        template_select = """\
                .dataframe %s {
                    %s: %s;
                }"""
        element_props = [
            ("tbody tr th:only-of-type", "vertical-align", "middle"),
            ("tbody tr th", "vertical-align", "top"),
        ]
        # if isinstance(self.columns, MultiIndex):
        #     element_props.append(("thead tr th", "text-align", "left"))
        #     if self.show_row_idx_names:
        #         element_props.append(
        #             ("thead tr:last-of-type th", "text-align", "right")
        #         )
        # else:
        #     element_props.append(("thead th", "text-align", "right"))
        element_props.append(("thead th", "text-align", "right"))

        template_mid = "\n\n".join(map(lambda t: template_select % t, element_props))
        template = dedent("\n".join((template_first, template_mid, template_last)))
        self.write(template)

    def render(self): # -> list[str]:
        self.write("<div>")
        self.write_style()
        super().render()
        self.write("</div>")
        return self.elements

class TidyDataFrameRenderer(DataFrameRenderer):
    def to_html(
        self,
        buf = None,
        encoding = None,
        classes = None,
        notebook = False,
        border = None,
        table_id = None,
        render_links = False,
    ): # -> str | None:
        # from pandas.io.formats.html import (
        #     HTMLFormatter,
        #     NotebookFormatter,
        # )

        Klass = TidyNotebookFormatter if notebook else TidyHTMLFormatter

        html_formatter = Klass(
            self.fmt,
            classes=classes,
            border=border,
            table_id=table_id,
            render_links=render_links,
        )
        string = html_formatter.to_string()
        return save_to_buffer(string, buf=buf, encoding=encoding)




