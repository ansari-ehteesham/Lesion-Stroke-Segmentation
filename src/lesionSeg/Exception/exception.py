import sys
import inspect
from ensure import EnsureError


# Normal Error Handling
def error_message_detail(error):
    stack = inspect.stack()
    # The third entry in the stack is the caller of the exception
    if len(stack) >= 3:
        frame_info = stack[2]
        file_name = frame_info.filename
        line_number = frame_info.lineno
    else:
        file_name = "unknown"
        line_number = 0


    error_message = f"File Name: {file_name}\nLine Number: {line_number}\nError: {error}"

    return error_message

class CustomeException(Exception):
    def __init__(self, error_message):
        super().__init__(error_message)
        self.error_message = error_message_detail(error=error_message)

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