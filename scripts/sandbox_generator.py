import duckdb
import os

# this function creates an analytical sandbox,
# consisting in filtering some house types and/or states
# inputs:
#   - types, states: lists of types and states to filter
#   - database_path: path of the DBMS
#   - table_name: optional
def createSandbox(types=[], states=[], database_path = 'data/exploitation.db', table_name=""):

    # set the correct data types in case the input were not lists
    if type(types) == str:
        types = [types]
    elif type(types) == list:
        pass
    else: types = list(types)

    if type(states) == str:
        states = [states]
    elif type(states) == list:
        pass
    else: states = list(states)

    # create the query string
    query_where = []
    if len(types) > 0:
        query_where.append("type in (" + ",".join(["'"+x+"'" for x in types]) + ")")
    if len(states) > 0:
        query_where.append("state in (" + ",".join(["'"+x+"'" for x in states]) + ")")

    if len(query_where) > 0:
        query_where = " where " + " and ".join(query_where)
    else:
        query_where = ""

    query = "select * from houses" + query_where

    # create a descriptive name of the table to save the result
    # only in case that the user doesnt already input a name
    if table_name == "":
        table_name = "sandbox"

        if len(types)>0:
            table_name = table_name + "_T" + "".join([
                "_" + x.replace('/','').replace('-','').replace(' ','')
                for x in types ])

        if len(states)>0:
            table_name = table_name + "_S" + "".join([ "_" + x for x in states ])

    # apply the query
    con = duckdb.connect(database_path)
    con.execute(f"create or replace table {table_name} as ( {query} )")
    con.close()

    # return the name of the table created
    return table_name

def sandbox_generator_help(col="all") :
    
    if col=="state":
        print("valid states: 'ak', 'al', 'ar', 'az', 'ca', 'co', 'ct', 'dc', 'de', " +
        "'fl', 'ga', 'hi', 'ia', 'id', 'il', 'in', 'ks', 'ky', 'la', 'ma', 'md', " +
        "'me', 'mi', 'mn', 'mo', 'ms', 'mt', 'nc', 'nd', 'ne', 'nh', 'nj', 'nm', " +
        "'nv', 'ny', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', " +
        " 'va', 'vt', 'wa', 'wi', 'wv', 'wy'")
    elif col == "type":
        print("valid types: 'apartment', 'assisted living', 'condo', 'cottage/cabin', 'duplex', " +
        "'flat', 'house', 'in-law', 'land', 'loft', 'manufactured', 'townhouse'")
    else:
        print("valid keys: 'state', ' type'\n")
        sandbox_generator_help('state')
        print("")
        sandbox_generator_help('type')


if __name__ == "__main__":
    if os.getcwd().replace("\\", "/").split("/")[-1] in ["notebooks", "scripts"]:
        os.chdir("..")
    createSandbox("apartment", "ca")