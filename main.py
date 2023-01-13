from scripts.sandbox_generator import *
from scripts.feature_generation import *
from scripts.model import *

import duckdb

con = duckdb.connect('data/exploitation.db', read_only=True)
con.execute("show tables").df()
con.close()

# sandbox_generator_help()

sandboxName = createSandbox(
    types=["apartment"],
    states=["ca"],
    database_path='data/exploitation.db'
)

preprocessing(
    data=sandboxName,
    target="price",
    drop_features =["id", "url", "region_url", "image_url", "description"],
    skip_log=["beds","baths","lat","long"],
    skip_outliers=["lat","long"]
)

best_model(sandboxName)
print("Check folder 'output' for the plots of the model")

