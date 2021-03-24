#  Copyright 2019 Gabriele Valvano
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import tensorflow as tf
tf.random.set_seed(12345)
import numpy as np
np.random.seed(12345)
import random
random.seed(12345)
import sqlite3
from tabulate import tabulate
import config as run_config

args = run_config.define_flags()
database = 'test_results.db'


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file. If you pass the file name as ':memory:', it will create a new database that resides
                    in the memory (RAM) instead of a database file on disk.
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)

    return conn


def textToList(string_list):
    tmp_list = string_list.strip('[]').replace('\'', '').replace(' ', '').split(',')
    return [np.float(el) for el in tmp_list]


def main():

    # create a database connection
    conn = create_connection(database)
    table_name = args.table_name
    run_id = args.RUN_ID

    # read values from SQL database:
    table = []

    for perc in ['perc5', 'perc10', 'perc25']:
        try:
            sql_command = """SELECT COUNT(SPLIT), AVG(AVERAGE_DICE), AVG(STD_DICE)
                             FROM {0} 
                             WHERE PERC == '{1}' AND (RUN_ID LIKE '{2}_{1}_split%')""".format(table_name, perc, run_id)
            # print(sql_command)

            cur = conn.cursor()
            cur.execute(sql_command)
            rows = cur.fetchall()

            for row in rows:
                table.append([perc, row[0], row[1] * 100, row[2] * 100])
        except:
            # The record does not exist
            pass

    print('\n{0}\n({1}):\n{2}\n'.format('_' * 60, table_name, args.RUN_ID))
    print(tabulate(table, headers=['PERC', 'SPLIT', 'AVG(AVERAGE_DICE) %', 'AVG(STD_DICE) %']))
    print('\n')


if __name__ == '__main__':
    main()
