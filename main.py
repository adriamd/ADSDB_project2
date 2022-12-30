from sandbox_generator import *
from feature_generation import *
#from model import *

import duckdb

con = duckdb.connect('data/exploitation.db', read_only=True)
con.execute("show tables").df()
con.close()

sandbox_generator_help()

sandboxName = createSandbox(
    types=["apartment"],
    states=["ga", "fl"], # exemple random per provar, ja veurem que pillar al sandbox
    database_path='data/exploitation.db'
)

preprocessing(
    data=sandboxName,
    target="price",
    drop_features =["id", "url", "region_url", "image_url", "description"],
    skip_log=["beds","baths","lat","long"],
    skip_outliers=["lat","long"]
)


# models()

