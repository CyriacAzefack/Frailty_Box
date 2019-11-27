import csv
import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import dateutil.parser

DEBUG = True
brokerUrl = "https://broker.aesio.dev.omwave.me"

output = "./sensors.csv"


def toIso(datetime):
    if datetime.tzinfo is None:
        print("WARNING: datetime object {} has no timezone".format(datetime))
    return datetime.isoformat()


def get(path, params=None):
    requestUrl = brokerUrl + path
    if params is not None:
        requestUrl += "?" + urllib.parse.urlencode(params)
    if DEBUG:
        print("DEBUG: GET '{}'".format(requestUrl))
    with urllib.request.urlopen(requestUrl) as url:
        return json.loads(url.read().decode())


def getMessages(deviceAddr, fromDate=None, toDate=None):
    # note: fromDate is inclusive
    #       toDate is exclusive (down to the millisecond)
    params = {'device': deviceAddr}
    if fromDate is not None: params['from'] = toIso(fromDate)
    if toDate is not None: params['to'] = toIso(toDate)

    return get("/messages", params)


def getStatusMessages(deviceAddr, fromDate=None, toDate=None):
    """ List Status messages received from a device
    Important note: The dates passed in arguments refer to the dates at which the messages were received
                    by the server, not to the date at which they were created on the device.
                    The date returned is the date of creation on the device.
    """
    # print("INFO: Getting status messages for {} from '{}'' to '{}'".format(deviceAddr, fromDate, toDate))
    for m in getMessages(deviceAddr, fromDate, toDate):
        print(m)
        # print(m['packet']['control']['receive']['devicePacket'])
        control = m['packet']['control']
        if 'receive' in control:
            devicePacket = control['receive']['devicePacket']
            # print(devicePacket)
            if ('statusMessage' in devicePacket) and (devicePacket['statusMessage'] is not None):
                # print(devicePacket['statusMessage'])
                for (key, value) in devicePacket['statusMessage'].items():
                    if 'status' in key.lower():
                        value['date'] = m['when']
                        yield value


def getStatusMessagesForDevices(deviceAddrList, fromDate=None, toDate=None):
    print("INFO: Getting status messages for devices {} from '{}' to '{}'".format(deviceAddrList, fromDate, toDate))
    if (type(deviceAddrList) == str): raise TypeError("Argument deviceAddrList must be a list, not a string.")
    for deviceAddr in deviceAddrList:
        for message in getStatusMessages(deviceAddr, fromDate, toDate):
            yield (deviceAddr, message)


def getDevice(deviceAddr):
    return get("/devices/" + deviceAddr)


def getSubdevices(gatewayAddress):
    # return get("/devices", {"gatewayBluetoothAddr": gatewayAddress})
    all_devices = get("/devices")  # Get all devices

    # Filter the list
    devices_for_gateway = []
    for device in all_devices:
        if device['gatewayBluetoothAddr'] == gatewayAddress:
            devices_for_gateway.append(device)

    return devices_for_gateway


def getSubDevicesAddr(gatewayAddress):
    return [d['bluetoothAddr'] for d in getSubdevices(gatewayAddress)]


def getMessagesLoop(deviceAddrList, timestep, from_time=None):
    """
    This is a generator that loops indefinitely, yielding new elements as they are found
    """
    to_time = datetime.now(timezone.utc)

    # print("Begin Warming!!")
    # for i in trange(120, desc="Warming the system"):
    #     sleep(1)
    # print("End Warming!!")

    while True:
        if from_time is None: from_time = to_time

        try:
            for (d, m) in getStatusMessagesForDevices(deviceAddrList, from_time, to_time):
                yield (d, m)

            from_time = to_time
        except urllib.error.URLError as e:
            # if there was an error display it, from_time will not be modified so the missed data will be recovered at next interval
            print("ERROR: {}".format(e))

        # sleep(timestep.seconds)

        to_time += timestep

def printStatusMessage(deviceAddr, statusMessage):
    try:
        print('DEVICE_DATA: addr: {}, message: {}'.format(deviceAddr, statusMessage))
    except ValueError:
        pass


gatewayAddress = "D1:B4:D8:4F:DF:85"

print(f"GATEWAY : {getDevice(gatewayAddress)}")

subdevices = getSubdevices(gatewayAddress=gatewayAddress)
deviceAddrList = getSubDevicesAddr(gatewayAddress)
for subdevice in subdevices:
    print(f"\t{subdevice}")
# print(deviceAddrList)


# list messages from last day for this device
# deviceAddr = subdevices[1]['bluetoothAddr']
#
# deviceAddr = "FB:22:F0:83:70:F5"
# for m in getStatusMessages(deviceAddr, fromDate = datetime.now(timezone.utc) - timedelta(days=1)):
#     printStatusMessage(deviceAddr, m)

# # list all messages starting now, gathering them at 10 minutes interval 
# for (deviceAddr, m) in getMessagesLoop(deviceAddrList, timestep=timedelta(seconds=5)):
#     printStatusMessage(deviceAddr, m)

# # list all messages starting 1 hour ago, gathering them at 5 seconds interval

start_time = datetime.now(timezone.utc)
timestep = timedelta(seconds=10)

with open(output, 'w+', newline='') as csvfile:
    fieldnames = ['sensorAddr', 'date', 'battery', 'temperature', 'humidity', 'radarDetected', 'magnetDetected',
                  'accelerationDetected']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for (deviceAddr, m) in getMessagesLoop(deviceAddrList, timestep=timestep, from_time=start_time):
        row = m
        row['sensorAddr'] = deviceAddr
        date = row['date']
        row['date'] = dateutil.parser.parse(date).strftime("%Y-%m-%d %H:%M:%S.%f")
        writer.writerow(row)
        csvfile.flush()

        printStatusMessage(deviceAddr, m)
