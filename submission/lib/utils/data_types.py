class PlatformIndex():  # pylint: disable=too-few-public-methods
    '''Structure for platform indices'''

    __slots__ = ['item_id', 'name', 'bomc_id', 'timestamp', 'value', 'cmdb_id']

    def __init__(self, data):
        self.item_id = data['itemid']
        self.name = data['name']
        self.bomc_id = data['bomc_id']
        self.timestamp = data['timestamp'] 
        self.value = data['value']
        self.cmdb_id = data['cmdb_id']


class BusinessIndex():  # pylint: disable=too-few-public-methods
    '''Structure for business indices'''

    __slots__ = ['service_name', 'start_time', 'avg_time', 'num',
                 'succee_num', 'succee_rate']

    def __init__(self, data):
        self.service_name = data['serviceName']
        self.start_time = data['startTime']
        self.avg_time = data['avg_time']
        self.num = data['num']
        self.succee_num = data['succee_num']
        self.succee_rate = data['succee_rate']


class Trace():  # pylint: disable=invalid-name,too-many-instance-attributes,too-few-public-methods
    '''Structure for traces'''

    __slots__ = ['call_type', 'start_time', 'elapsed_time', 'success',
                 'trace_id', 'id', 'pid', 'cmdb_id', 'service_name', 'ds_name']

    def __init__(self, data):
        self.call_type = data['callType']
        self.start_time = data['startTime']
        self.elapsed_time = data['elapsedTime']
        self.success = data['success']
        self.trace_id = data['traceId']
        self.id = data['id']
        self.pid = data['pid']
        self.cmdb_id = data['cmdb_id']

        if 'serviceName' in data:
            # For data['callType']
            #  in ['CSF', 'OSB', 'RemoteProcess', 'FlyRemote', 'LOCAL']
            self.service_name = data['serviceName']
        if 'dsName' in data:
            # For data['callType'] in ['JDBC', 'LOCAL']
            self.ds_name = data['dsName']
            self.service_name = self.ds_name

    def __repr__(self):
        return f'{self.call_type}\t{self.start_time}\t{self.elapsed_time}\t{self.success}\t{self.trace_id}\t{self.id}\t{self.pid}\t{self.cmdb_id}\t{self.service_name}\t{self.ds_name}'
