import d3rlpy
try:
    config = d3rlpy.algos.IQLConfig(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        weight_temp=3.0
    )
    print("IQLConfig instantiation successful")
except TypeError as e:
    print(f"IQLConfig instantiation failed: {e}")
    # Try valid args
    try:
        config = d3rlpy.algos.IQLConfig(weight_temp=3.0)
        print("IQLConfig with only weight_temp successful")
    except Exception as e2:
        print(f"IQLConfig simple failed: {e2}")
