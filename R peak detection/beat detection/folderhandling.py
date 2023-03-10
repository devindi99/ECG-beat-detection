def mkdir_p(mypath: str):
    """

    :param mypath: path where the folder will be created
    :return:
    """

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise