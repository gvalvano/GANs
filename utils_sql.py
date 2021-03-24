import sqlite3
import datetime


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


def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except sqlite3.Error as e:
        print(e)


def alter_table(conn, alter_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param alter_table_sql: a ALTER TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(alter_table_sql)
    except sqlite3.Error as e:
        print(e)


def insert_new_record(conn, table_name, values):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param table_name: name of the table in the database
    :param values: values to insert
    :return:
    """
    sql = """INSERT INTO {0}(RUN_ID, PERC, SPLIT, CONFIG, EXPERIMENT_TYPE, DATASET_NAME, INPUT_SIZE, EPOCH,
            AVERAGE_DICE, STD_DICE, AVERAGE_DICE_PER_CLASS, STD_DICE_PER_CLASS, DICE_VALUES, DICE_VALUES_PER_CLASS, 
            AVERAGE_IOU, STD_IOU, AVERAGE_IOU_PER_CLASS, STD_IOU_PER_CLASS, IOU_VALUES, IOU_VALUES_PER_CLASS, TIMESTAMP) 
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""".format(table_name)
    cur = conn.cursor()
    time_stamp = datetime.datetime.now().strftime("%Y-%b-%d, %A %I:%M:%S")
    values = values + [time_stamp]
    cur.execute(sql, values)
    return cur.lastrowid  # return the generated id


def add_db_entry(entries, table_name, database='test_results_by_patient.db'):
    """Insert values in results SQL database. """
    # create a database connection
    conn = create_connection(database)
    # table_name = args.table_name

    # create tables
    if conn is not None:
        statement = """ CREATE TABLE IF NOT EXISTS {0} (
                        id integer PRIMARY KEY,
                        RUN_ID text NOT NULL,
                        PERC text NOT NULL,
                        SPLIT text NOT NULL,
                        CONFIG text NOT NULL,
                        EXPERIMENT_TYPE text NOT NULL,
                        DATASET_NAME text NOT NULL,
                        INPUT_SIZE text NOT NULL,
                        EPOCH integer NOT NULL,
                        AVERAGE_DICE float NOT NULL,
                        STD_DICE float NOT NULL,
                        AVERAGE_DICE_PER_CLASS text NOT NULL,
                        STD_DICE_PER_CLASS text NOT NULL,
                        DICE_VALUES text NOT NULL,
                        DICE_VALUES_PER_CLASS text NOT NULL,
                        AVERAGE_IOU float NOT NULL,
                        STD_IOU float NOT NULL,
                        AVERAGE_IOU_PER_CLASS text NOT NULL,
                        STD_IOU_PER_CLASS text NOT NULL,
                        IOU_VALUES text NOT NULL,
                        IOU_VALUES_PER_CLASS text NOT NULL,
                        TIMESTAMP text
                    ); """.format(table_name)
        create_table(conn, statement)
    else:
        print("\033[31m  Error! cannot create the database connection.\033[0m")
        raise

    with conn:
        # get values and insert into table:
        values = [entries['run_id'], entries['n_sup_vols'], entries['split_number'], str(entries['config']),
                  entries['experiment_type'], entries['dataset_name'],
                  str(entries['input_size']),
                  int(entries['epoch']),
                  float(entries['avg_dice']),
                  float(entries['std_dice']),
                  str(entries['avg_dice_per_class']),
                  str(entries['std_dice_per_class']),
                  str(entries['dice_list']),
                  str(entries['dice_list_per_class']),
                  float(entries['avg_iou']),
                  float(entries['std_iou']),
                  str(entries['avg_iou_per_class']),
                  str(entries['std_iou_per_class']),
                  str(entries['iou_list']),
                  str(entries['iou_list_per_class'])
                  ]
        insert_new_record(conn, table_name, values)
