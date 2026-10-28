class TaskError(Exception):
    """A processing failure reported to the client as `error_fe`.

    Codes the frontend understands: password, corrupt_pdf,
    no_pages_in_pdf, too_many_pages, page_too_large, no_text_found, input_tokens,
    invalid_options, internal_server_error; cancelled, abandoned (app/tasks.py: the page
    was closed or stopped asking).
    """

    def __init__(self, code: str, detail: str = ""):
        super().__init__(detail or code)
        self.code = code
        self.detail = detail
