import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from time import sleep

brokerUrl = "https://broker.aesio.dev.omwave.me"


def toIso(datetime):
    if datetime.tzinfo is None:
        print("WARNING: datetime object {} has no timezone".format(datetime))
    return datetime.isoformat()


def get(path, params=None):
    requestUrl = brokerUrl + path
    if params is not None:
        requestUrl += "?" + urllib.parse.urlencode(params)
    print("DEBUG: GET '{}'".format(requestUrl))
    with urllib.request.urlopen(requestUrl) as url:
        return json.loads(url.read().decode())


def getMessages(deviceAddr, fromDate=None, toDate=None):
    # note: fromDate is inclusive
    #       toDate is exclusive (down to the millisecond)
    params = {'device': deviceAddr}
    if (fromDate != None): params['from'] = toIso(fromDate)
    if (toDate != None): params['to'] = toIso(toDate)
    return get("/messages", params)


def getStatusMessages(deviceAddr, fromDate=None, toDate=None):
    """ List Status messages received from a device
    Important note: The dates passed in arguments refer to the dates at which the messages were received
                    by the server, not to the date at which they were created on the device.
                    The date returned is the date of creation on the device.
    """
    # print("INFO: Getting status messages for {} from '{}'' to '{}'".format(deviceAddr, fromDate, toDate))
    for m in getMessages(deviceAddr, fromDate, toDate):
        # print(m)
        # print(m['packet']['control']['receive']['devicePacket'])
        control = m['packet']['control']
        if 'receive' in control:
            devicePacket = control['receive']['devicePacket']
            # print(devicePacket)
            if 'statusMessage' in devicePacket and devicePacket['statusMessage'] is not None:
                # print(devicePacket['statusMessage'])
                for (key, value) in devicePacket['statusMessage'].items():
                    if 'status' in key.lower():
                        yield value


def getStatusMessagesForDevices(deviceAddrList, fromDate=None, toDate=None):
    print("INFO: Getting status messages for devices {} from '{}'' to '{}'".format(deviceAddrList, fromDate, toDate))
    if (type(deviceAddrList) == str): raise TypeError("Argument deviceAddrList must be a list, not a string.")
    for deviceAddr in deviceAddrList:
        for message in getStatusMessages(deviceAddr, fromDate, toDate):
            yield (deviceAddr, message)


def getDevice(deviceAddr):
    return get("/devices/" + deviceAddr)


def getDevicesForGateway(gatewayAddress):
    return get("/devices", {"gatewayBluetoothAddr": gatewayAddress})


def getDevicesAddrForGateway(gatewayAddress):
    return [d['bluetoothAddr'] for d in getDevicesForGateway(gatewayAddress)]


def getMessagesLoop(deviceAddrList, timestep, from_time=None):
    """
    This is a generator that loops indefinitely, yielding new elements as they are found
    """
    while True:
        to_time = datetime.now(timezone.utc)
        if from_time == None: from_time = to_time

        try:
            for (d, m) in getStatusMessagesForDevices(deviceAddrList, from_time, to_time):
                yield (d, m)

            from_time = to_time
        except urllib.error.URLError as e:
            # if there was an error display it, from_time will not be modified so the missed data will be recovered at next interval
            print("ERROR: {}".format(e))

        sleep(timestep.seconds)


def printStatusMessage(deviceAddr, statusMessage):
    try:
        print('DEVICE_DATA: addr: {}, message: {}'.format(deviceAddr, statusMessage))
    except ValueError:
        pass


gatewayAddress = "FE:2B:50:E1:87:D7"

deviceAddrList = getDevicesAddrForGateway(gatewayAddress)
print(deviceAddrList)

# print(getDevice(deviceAddr))

# # list messages from last day for this device
# deviceAddr = gatewayAddress
# for m in getStatusMessages(deviceAddr, fromDate = datetime.now(timezone.utc) - timedelta(days=1)):
#     printStatusMessage(deviceAddr, m)

# # list all messages starting now, gathering them at 10 minutes interval 
# for (deviceAddr, m) in getMessagesLoop(deviceAddrList, timedelta(minutes=10)):
#     printStatusMessage(deviceAddr, m)

# # list all messages starting 1 hour ago, gathering them at 5 seconds interval 
for (deviceAddr, m) in getMessagesLoop(deviceAddrList, timedelta(seconds=5),
                                       from_time=datetime.now(timezone.utc) - timedelta(hours=1)):
    printStatusMessage(deviceAddr, m)
