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
    def _truncate_horizontally(self) -> None:
        """Remove columns, which are not to be displayed and adjust formatters.
        Attributes affected:
            - tr_frame
            - formatters
            - tr_col_num
        """
        assert self.max_cols_fitted is not None
        # col_num = self.max_cols_fitted // 2
        col_num = self.max_cols_fitted
        if col_num >= 1:
            # left = self.tr_frame.iloc[:, :col_num]
            # right = self.tr_frame.iloc[:, -col_num:]
            # self.tr_frame = concat((left, right), axis=1)
            self.tr_frame = self.tr_frame.iloc[:, :col_num]

            # truncate formatter
            if isinstance(self.formatters, (list, tuple)):
                # self.formatters = [
                #     *self.formatters[:col_num],
                #     *self.formatters[-col_num:],
                # ]
                self.formatters = self.formatters[:col_num]
                    
        else:
            col_num = cast(int, self.max_cols)
            self.tr_frame = self.tr_frame.iloc[:, :col_num]
        self.tr_col_num = col_num

    def _truncate_vertically(self) -> None:
        """Remove rows, which are not to be displayed.
        Attributes affected:
            - tr_frame
            - tr_row_num
        """
        assert self.max_rows_fitted is not None
        # row_num = self.max_rows_fitted // 2
        row_num = self.max_rows_fitted
        if row_num >= 1:
            # head = self.tr_frame.iloc[:row_num, :]
            # tail = self.tr_frame.iloc[-row_num:, :]
            # self.tr_frame = concat((head, tail))
            self.tr_frame = self.tr_frame.iloc[:row_num, :]
        else:
            row_num = cast(int, self.max_rows)
            self.tr_frame = self.tr_frame.iloc[:row_num, :]
        self.tr_row_num = row_num

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

    def render(self): # -> list[str]:
        by = chr(215)  # ×
        # font-style:oblique;
        self.write(
            f"<p style='color:#606060;font-family:verdana;font-size:10px'> A tidy dataframe: {len(self.frame)} {by} {len(self.frame.columns)} </p>"
        )
        self._write_table()

        # if self.should_show_dimensions:
        #     by = chr(215)  # ×
        #     self.write(
        #         f"<p>{len(self.frame)} rows {by} {len(self.frame.columns)} columns</p>"
        #     )

        footer_str = ""
        if self.fmt.is_truncated_vertically:
            footer_str += "#... with {} more rows".format(len(self.frame)-self.fmt.tr_row_num)

        max_footer_cols_print = 100
        if self.fmt.is_truncated_horizontally and self.fmt.tr_col_num < len(self.frame.columns):
            more_cols = len(self.frame.columns) - self.fmt.tr_col_num
            footer_cols = ["{} &lt{}&gt".format(cname, self.frame[cname].dtype.name) 
                              for cname in self.frame.columns[self.fmt.tr_col_num:]
                          ]
            col_footer_str = "{} more columns: {}".format(more_cols, ", ".join(footer_cols[0:max_footer_cols_print]))
            
            if more_cols > max_footer_cols_print:
                col_footer_str += "..."
            if self.fmt.is_truncated_vertically:
                footer_str += ", and " + col_footer_str
            else:
                footer_str += "#... with " + col_footer_str

        self.write(f"<p style='color:#606060;font-family:verdana;font-size:10px'>{footer_str}</p>")

        return self.elements

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




