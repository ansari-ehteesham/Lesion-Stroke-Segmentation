import sys
from ensure import EnsureError


# Normal Error Handling
def error_message_detail(error, error_detail: sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno


    error_message = f"File Name: {file_name}\nLine Number: {line_number}\nError: {error}"

    return error_message

class CustomeException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error=error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
    



# Ensure Error Handling
def catch_ensure_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except EnsureError as e:
            raise CustomeException(e,sys)
    return wrapper