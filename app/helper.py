import json
import pandas as pd

TRAINED_DATA = []


async def read_file(file_path: str) -> json:
    try:
        data = pd.read_csv(file_path, low_memory=False)
        data_dict = data.to_dict(orient="records")
        return json.dumps({"status": 200, "content": data_dict})
    except FileNotFoundError:
        # The case where the file doesn't exist
        return json.dumps({"status": 501, "message":
                           f"The file '{file_path}' was not found."})
    except PermissionError:
        # The case where the file can't be accessed due to permission issues
        return json.dumps({"status": 404, "message":
                           f"You do not have permission to read the file '{file_path}'."})
    except Exception as e:
        # Any other exceptions
        return json.dumps({"status": 501,
                           "message": f"An error occurred while reading the file: {e}"})


async def modify_global(value: str):
    global TRAINED_DATA
    TRAINED_DATA.append(value)


async def remove_global():
    global TRAINED_DATA
    TRAINED_DATA.clear()
