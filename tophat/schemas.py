factors_avro = {
    'namespace': 'com.gilt.cerebro.job',
    'type': 'record',
    'name': 'AvroFactors',
    'fields': [
        {'name': 'id', 'type': 'string'},
        {'name': 'factors', 'type': {'type': 'array',
                                     'items': 'float'}},
        {'name': 'bias', 'type': 'float'},
    ],
}

