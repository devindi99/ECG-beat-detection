def mkdir_p(mypath: str) -> None:
    """

    :param mypath: path where the folder will be created
    :return: None
    """

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise

    return None
