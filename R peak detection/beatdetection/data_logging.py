import openpyxl


def log_data(
        data: tuple,
        wb: str,
        sh: str):
    workbook = openpyxl.load_workbook(wb)
    sheet = workbook[sh]
    sheet.append(data)

    return None

