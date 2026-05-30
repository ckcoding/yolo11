custom_imports = dict(
    allow_failed_imports=False, imports=[
        'mock_module',
    ])
train = dict(collate=dict(type='SafeCollateClass'))
