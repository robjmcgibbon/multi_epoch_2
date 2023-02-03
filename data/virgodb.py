#!/bin/env python
#
# Module to access VirgoDB database via the web interface.
#
# This can be used to access the Millennium and EAGLE tables,
# for example.
#

# Pandas read_csv is faster than numpy loadtxt, so use if available
try:
    import pandas
except ImportError:
    pandas = None

# Can write HDF5 output if we have h5py
try:
    import h5py
except ImportError:
    h5py = None

import numpy as np
import urllib.request, urllib.parse, urllib.error
import urllib.request, urllib.error, urllib.parse
import http.cookiejar
import re
import shutil
from getpass import getpass
import io
import sys

# Chunk size for HDF5 output
hdf5_chunk_size = 65536

class VirgoDBError(Exception):
    """Base class for exceptions raised by this module"""
    pass

class SQLError(VirgoDBError):
    """Exception raised if an SQL query fails"""
    pass

class BadResponseError(VirgoDBError):
    """Exception raised if we can't interpret the result of a query"""
    pass


# Convert received text to a numpy record array, using pandas if available.
def text_to_array(f, dtype):
    if pandas is None:
        return  np.loadtxt(f, dtype=dtype, delimiter=",")
    else:
        dtypes = {name : dtype.fields[name][0] for name in dtype.names}
        return pandas.read_csv(f, names=dtype.names, dtype=dtypes, delimiter=",", engine="c").to_records(index=False)


# Mapping between SQL and numpy types
numpy_dtype = {
    "real"     : np.float32,
    "float"    : np.float64,
    "int"      : np.int32,
    "bigint"   : np.int64,
    "char"     : np.dtype("|S256"),
    "nvarchar" : np.dtype("|S256"),
    "decimal"  : np.float64
    }


# Default database URL
default_url = "http://virgodb.dur.ac.uk:8080/MyMillennium"


# Cookie storage - want to avoid creating a new session for every query
cookie_file = "sql_cookies.txt"
cookie_jar = http.cookiejar.LWPCookieJar(cookie_file)
try:
    cookie_jar.load(ignore_discard=True)
except IOError:
    pass

class VirgoDB:

    def __init__(self, username, password=None, url=default_url):
        """
        Class to store info required to connect to the web server

        Parameters:

        username: user to log in as
        password: user's password - will prompt if not supplied
        url     : database web interface URL
        """
        # Get password if necessary
        if password is None:
            password = getpass()
        # Get URL for the database
        self.db_url = url
        # Set up authentication and cookies
        self.password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        self.password_mgr.add_password(None, self.db_url, username, password)
        self.opener = urllib.request.OpenerDirector()
        self.auth_handler   = urllib.request.HTTPBasicAuthHandler(self.password_mgr)
        self.cookie_handler = urllib.request.HTTPCookieProcessor(cookie_jar)


    def _check_response_ok(self, response):
        
        # Check for OK response
        line = response.readline()
        if line != "#OK\n":
            # Try to convert the error message into something readable!
            error_string = None
            for line in response.readlines():
                if line.decode('utf-8').startswith(("#SQLEXCEPTION")):
                    error_string = line[15:]
            if error_string is not None:
                raise SQLError(error_string.strip())
            # else:
                # raise SQLError(response.readlines)        
                # response.readlines
 
    def _read_column_info(self, response):
        
        # Skip rows until we reach QUERYTIMEOUT
        while True:
            line = response.readline()
            line = line.decode('utf-8')
            if line == "":
                raise BadResponseError("Unexpected end of file while reading result header")
            if line.startswith(("#QUERYTIMEOUT")):
                break

        # Skip QUERYTIME
        if not((response.readline()).decode('utf-8').startswith(("#QUERYTIME"))):
            raise BadResponseError("Don't understand result header!")

        # Read column info
        # (also discards line with full list of column names)
        columns = []
        while True:
            line = response.readline()
            line = line.decode('utf-8')
            if line[0] != "#":
                break
            else:
                m = re.match("^#COLUMN ([0-9]+) name=([\w]+) JDBC_TYPE=(-?[0-9]+) JDBC_TYPENAME=([\w]+)$", line)
                if m is not None:
                    columns.append(m.groups())
                else:
                    raise BadResponseError("Don't understand column info: "+line)
        
        return columns
    

    def execute_query(self, sql):
        """
        Run an SQL query and return the result as a record array.

        Parameters:

        sql: a string with the SQL query

        Returns a record array where each field is a column of the
        table returned by the database. Each element of the array
        is one row.

        Note that this can be quite CPU intensive due to the conversion
        from text to a numpy array. Performance is better if pandas is
        available.
        """
        url = self.db_url + "?" + urllib.parse.urlencode({'action': 'doQuery', 'SQL': sql})
        urllib.request.install_opener(urllib.request.build_opener(self.auth_handler, self.cookie_handler))
        response = urllib.request.urlopen(url)
        cookie_jar.save(ignore_discard=True)

        # Read header with columns
        # print(response)
        try:
            columns = self._read_column_info(response)
        except:
            # Check for OK response
            self._check_response_ok(response)

        # Construct record type for the output
        dtype = np.dtype([(col[1],numpy_dtype[col[3]]) for col in columns])

        # Return the data as a record array
        return text_to_array(response, dtype)


    def query_to_file(self, outfile, sql, format="text"):
        """
        Run an SQL query and stream the result to a file.

        Parameters:

        outfile: name of the file to write
        sql    : string with the SQL query
        format : output file format - 'text' or 'hdf5'

        Writing a text file is the fastest way to save large
        result sets.

        HDF5 files are written as one 1D dataset for each
        column in the query result.
        """
        url = self.db_url + "?" + urllib.parse.urlencode({'action': 'doQuery', 'SQL': sql})
        urllib.request.install_opener(urllib.request.build_opener(self.auth_handler, self.cookie_handler))
        response = urllib.request.urlopen(url)
        cookie_jar.save(ignore_discard=True)

        # Check for OK response
        self._check_response_ok(response)

        if format == 'text':

            # Write remainder of response to a file as text
            out = open(outfile, "w")
            shutil.copyfileobj(response, out)
            out.close()

        elif format == 'hdf5':

            # Check we have h5py
            if h5py is None:
                raise ImportError("HDF5 output requested but could not import h5py")

            # Get column information
            columns = self._read_column_info(response)
            names   = [col[1] for col in columns]
            dtypes  = [numpy_dtype[col[3]] for col in columns]
            rectype = np.dtype([(col[1],numpy_dtype[col[3]]) for col in columns])

            # Create the output file and datasets
            with h5py.File(outfile, "w") as out:
                for name, dtype in zip(names, dtypes):
                    out.create_dataset(name, dtype=dtype, shape=(0,), maxshape=(None,), 
                                       chunks=(hdf5_chunk_size,))

                # Read results and append to output file
                nwritten = 0
                while True:

                    # Read some rows
                    lines = response.readlines(hdf5_chunk_size)
                    if len(lines)==0:
                        break

                    # Convert these rows to a record array
                    data = text_to_array(io.StringIO("".join(lines)), rectype)

                    # Append rows to the output file
                    for name in names:
                        dataset = out[name]
                        dataset.resize((nwritten+data.shape[0],))
                        dataset[nwritten:nwritten+data.shape[0]] = data[name]
                    nwritten += data.shape[0]

        else:
            raise ValueError("Format parameter must be 'text' or 'hdf5'!")


def connect(user, password=None, url=default_url):
    """Connect to database and return a connection object"""
    return VirgoDB(user, password, url=url)


def execute_query(con, sql):
    return con.execute_query(sql)

