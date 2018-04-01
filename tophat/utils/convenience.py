# TODO: Replica of lightfm-shenanigans/convenience

import pickle
import pandas as pd
import os
import fastavro as avro
import io, bz2, gzip
import shutil
import functools
from lib_cerebro_py.aws.aws_s3_uri import AwsS3Uri
from lib_cerebro_py.aws.aws_s3_object import AwsS3Object
from tophat.utils.log import logger

# TODO: move some methods to something like custom_io


def with_extension(extension, path):
    return extension in path.split('.', 1)[-1]


def open_fun_via_path(path, compression='infer'):
    """Determines which open function to use
    by inferring the compression scheme from path extension or explicitly passing
    """
    # checking if path is really a string
    if compression == 'infer':
        _, ext = os.path.splitext(path)
        ext = ext.split('.')[-1]
    else:
        ext = compression

    if ext == 'bz2':
        return bz2.open
    elif ext == 'gz':
        return gzip.open
    else:
        return io.open


def decompress_fun_via_path(path, compression='infer'):
    """Determines which decompression function to use
    by inferring the compression scheme from path extension or explicitly passing
    """
    if compression == 'infer':
        _, ext = os.path.splitext(path)
        ext = ext.split('.')[-1]
    else:
        ext = compression

    if ext == 'bz2':
        return bz2.decompress
    elif ext == 'gz':
        return gzip.decompress
    else:
        return lambda x: x


def compress_fun_via_path(path, compression='infer'):
    """Determines which compression function to use
    by inferring the compression scheme from path extension or explicitly passing
    """
    if compression == 'infer':
        _, ext = os.path.splitext(path)
        ext = ext.split('.')[-1]
    else:
        ext = compression

    if ext == 'bz2':
        return bz2.compress
    elif ext == 'gz':
        return gzip.compress
    else:
        return lambda x: x


def read_fun_via_path(path, file_format='infer'):
    """Determines which read function to use
    by inferring from path extension or explicitly passing
    """
    if file_format == 'infer':
        _, ext = os.path.splitext(path)
        ext = ext.split('.')[-1]
    else:
        ext = file_format
    if ext == 'bz2_pickle':
        return functools.partial(import_pickle, compression='bz2')
    elif ext == 'msg':
        return pd.read_msgpack
    elif ext == 'csv' or ext == 'txt':
        # not working with s3 urls, needed for Saks stuff
        if AwsS3Uri.is_valid(path):
            def read_csv_from_bytes(csv_bytes, **kwargs):
                csv_as_text = io.StringIO(csv_bytes.decode('utf-8'))
                return pd.read_csv(csv_as_text, **kwargs)
            return read_csv_from_bytes
        else:
            return pd.read_csv
    elif ext == 'avro':
        return read_avro_to_df
    else:
        logger.warn('Cannot infer read function from path extension... defaulting to msgpack')
        return pd.read_msgpack


def export_pickle(obj, path, use_disk=False, compression='infer'):
    """
    Dumps to s3 if it's an s3 path, else just save to the given local path
    use_disk : If `True`, save temporary file locally first and then upload
    compress : Infer compression from extension (bz2 gz)
    """
    path_tmp_file = '/tmp/s3_upload'
    open_fun = open_fun_via_path(path, compression=compression)
    compress_fun = compress_fun_via_path(path, compression=compression)
    if AwsS3Uri.is_valid(path):
        if use_disk:
            pickle.dump(obj, open_fun(path_tmp_file, 'wb'), protocol=2)
            data = open_fun(path_tmp_file, 'rb')
        else:
            data = io.BytesIO(compress_fun(pickle.dumps(obj, protocol=2)))

        AwsS3Object(path).upload_file_object(data)
        # aws_helper.dump_to_s3(path, data)
    else:
        pickle.dump(obj, open_fun(path, 'wb'), protocol=2)


def import_pickle(path, compression='infer'):
    open_fun = open_fun_via_path(path, compression=compression)
    decompress_fun = decompress_fun_via_path(path, compression=compression)
    # TODO to be refactored
    if is_string(path):
        if AwsS3Uri.is_valid(path):
            with AwsS3Object(path).download_file_object() as data:
                return pickle.loads(decompress_fun(data.read()))
        else:
            return pickle.load(open_fun(path, 'rb'))
    else:
        return pickle.loads(decompress_fun(path))


def is_string(s): return isinstance(s, str)


def backup_file(path):
    p1, p2 = os.path.split(path)
    path_backup = os.path.join(p1, 'prev-' + p2)
    if AwsS3Uri.is_valid(path_backup):
        AwsS3Object(path).copy(path_backup)
    elif os.path.exists(path):
        shutil.copy(path, path_backup)
    else:
        logger.warn(f'{path} not found')

    logger.info('{} backed up to: {}'.format(path, path_backup))


def read_avro_to_df(path_or_buf):
    if isinstance(path_or_buf, str):
        buf = open(path_or_buf, 'rb')
    else:
        buf = path_or_buf

    reader = avro.reader(buf)
    df = pd.DataFrame(list(reader))
    try:
        buf.close()
    except AttributeError:
        logger.warn('Stream has not attribute close')
    return df


def filter_col_isin(df, col, inclusion_set):
    if len(inclusion_set) and col in df.columns:
        logger.info('Filtering on {} in {}'.format(col, inclusion_set))
        return df.loc[df[col].isin(inclusion_set)]
    else:
        logger.info('Nothing to filter on')
        return df


def log_shape_or_npartitions(df, name: str ='') -> None:
    """
    df : dataframe to log shape or npartitions
    name : optional name of dataframe as extra info
    """
    if hasattr(df, 'compute'):  # if dask dataframe
        logger.info(f'{name} npartitions:\t({df.npartitions})')
    else:
        logger.info(f'{name} shape:\t(%d,%d)' % df.shape)
