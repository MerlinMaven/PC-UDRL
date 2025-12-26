import d3rlpy
import inspect

print("d3rlpy version:", d3rlpy.__version__)
# Create a dummy algo to check fit
try:
    cql = d3rlpy.algos.CQLConfig().create(device="cpu")
    print("fit args:", inspect.signature(cql.fit))
except Exception as e:
    print(e)
