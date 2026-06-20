import sqlite3
from pathlib import Path

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

DATA = {
    2024: {
        "All Crossings": {
            "Automobiles": [8415266, 8311615, 9223498, 9107817, 9721620, 9697277, 9753988, 9776034, 9328932, 9619698, 9208864, 9536454, 111701063],
            "Buses": [169828, 163390, 177724, 178981, 184406, 173922, 183414, 181581, 171190, 188169, 168381, 172791, 2113777],
            "Trucks": [688096, 657445, 699126, 722287, 752825, 718993, 746468, 734833, 695333, 766322, 707192, 720683, 8609603],
            "Total Vehicles": [9273190, 9132450, 10100348, 10009085, 10658851, 10590192, 10683870, 10692448, 10195455, 10574189, 10084437, 10429928, 122424443],
            "E-ZPass Usage (%)": [89.2, 89.3, 89.2, 89.3, 89.3, 88.8, 88.5, 88.7, 89.0, 89.1, 88.9, 88.7, 89.0],
        },
        "George Washington Bridge": {
            "Automobiles": [3476222, 3412731, 3760713, 3724379, 3934990, 3935993, 4015769, 3950736, 3734676, 3899897, 3775113, 3899700, 45520919],
            "Buses": [19365, 19199, 20836, 21509, 23334, 21994, 24123, 23687, 22886, 24676, 21160, 19944, 262713],
            "Trucks": [366016, 347575, 374129, 386104, 397223, 377218, 387806, 370676, 342471, 380787, 356688, 371129, 4457822],
            "Total Vehicles": [3861603, 3779505, 4155678, 4131992, 4355547, 4335205, 4427698, 4345099, 4100033, 4305360, 4152961, 4290773, 50241454],
            "E-ZPass Usage (%)": [87.5, 87.6, 87.3, 87.5, 87.6, 87.1, 86.8, 87.0, 87.0, 87.2, 87.2, 87.0, 87.2],
        },
        "Lincoln Tunnel": {
            "Automobiles": [1225543, 1217559, 1368943, 1345618, 1441254, 1399173, 1363933, 1410989, 1399025, 1436004, 1324217, 1348481, 16280739],
            "Buses": [135235, 128853, 140030, 140382, 142727, 134165, 140817, 140197, 131861, 145540, 131560, 136784, 1648151],
            "Trucks": [87590, 82750, 87330, 91284, 96363, 89835, 91026, 95223, 93684, 101158, 88529, 88557, 1093329],
            "Total Vehicles": [1448368, 1429162, 1596303, 1577284, 1680344, 1623173, 1595776, 1646409, 1624570, 1682702, 1544306, 1573822, 19022219],
            "E-ZPass Usage (%)": [90.9, 91.3, 91.1, 91.3, 91.2, 90.6, 90.2, 90.0, 90.7, 90.9, 90.7, 90.4, 90.8],
        },
        "Holland Tunnel": {
            "Automobiles": [1167603, 1155517, 1282671, 1252696, 1321920, 1297693, 1292870, 1322009, 1286285, 1327274, 1265722, 1327734, 15299994],
            "Buses": [3748, 3667, 4443, 4687, 5110, 5230, 5270, 4959, 5126, 5327, 4428, 4257, 56252],
            "Trucks": [37194, 34733, 37191, 37398, 40096, 37619, 38840, 39594, 38383, 41177, 36896, 37644, 456765],
            "Total Vehicles": [1208545, 1193917, 1324305, 1294781, 1367126, 1340542, 1336980, 1366562, 1329794, 1373778, 1307046, 1369635, 15813011],
            "E-ZPass Usage (%)": [88.7, 88.8, 88.6, 88.6, 88.5, 88.0, 87.5, 87.5, 88.0, 88.3, 88.3, 88.2, 88.2],
        },
        "Goethals Bridge": {
            "Automobiles": [1226564, 1215077, 1328027, 1330228, 1448068, 1449078, 1472209, 1461604, 1333418, 1400313, 1342680, 1401223, 16408489],
            "Buses": [8541, 8588, 9029, 8971, 9547, 9064, 9592, 9245, 8174, 9218, 8045, 8536, 106550],
            "Trucks": [124914, 121452, 127549, 133198, 139420, 137820, 148153, 147299, 141548, 159252, 148309, 148725, 1677639],
            "Total Vehicles": [1360019, 1345117, 1464605, 1472397, 1597035, 1595962, 1629954, 1618148, 1483140, 1568783, 1499034, 1558484, 18192678],
            "E-ZPass Usage (%)": [89.5, 89.5, 89.5, 89.6, 89.4, 89.1, 88.9, 89.1, 89.5, 89.5, 89.2, 89.1, 89.3],
        },
        "Outerbridge Crossing": {
            "Automobiles": [1046228, 1041623, 1171763, 1144559, 1225514, 1255809, 1266863, 1280563, 1213874, 1205789, 1163430, 1215326, 14231341],
            "Buses": [1528, 1562, 1691, 1672, 1876, 1824, 1949, 1856, 1508, 1440, 1392, 1458, 19756],
            "Trucks": [55291, 53816, 56016, 56289, 59665, 55939, 60389, 62545, 59869, 61955, 56970, 55317, 694061],
            "Total Vehicles": [1103047, 1097001, 1229470, 1202520, 1287055, 1313572, 1329201, 1344964, 1275251, 1269184, 1221792, 1272101, 14945158],
            "E-ZPass Usage (%)": [93.3, 93.2, 93.3, 93.3, 93.2, 93.2, 93.0, 93.0, 93.3, 93.4, 93.2, 93.0, 93.2],
        },
        "Bayonne Bridge": {
            "Automobiles": [273106, 269108, 311381, 310337, 349874, 359531, 342344, 350133, 361654, 350421, 337702, 343990, 3959581],
            "Buses": [1411, 1521, 1695, 1760, 1812, 1645, 1663, 1637, 1635, 1968, 1796, 1812, 20355],
            "Trucks": [17091, 17119, 16911, 18014, 20058, 20562, 20254, 19496, 19378, 21993, 19800, 19311, 229987],
            "Total Vehicles": [291608, 287748, 329987, 330111, 371744, 381738, 364261, 371266, 382667, 374382, 359298, 365113, 4209923],
            "E-ZPass Usage (%)": [89.1, 88.9, 88.9, 89.2, 89.2, 88.5, 88.4, 88.5, 88.8, 88.9, 88.5, 88.3, 88.7],
        },
    },
    2025: {
        "All Crossings": {
            "Automobiles": [8340612, 7732875, 9147352, 9055022, 9768957, 9541381, 9783173, 9992425, 9302537, 9464967, 9218011, 9239845, 110587157],
            "Buses": [170184, 160039, 176223, 179028, 182514, 177116, 181895, 176785, 168573, 179150, 168115, 180586, 2100208],
            "Trucks": [709686, 651999, 725833, 738349, 753133, 732464, 768002, 742405, 749938, 778176, 697403, 747601, 8794989],
            "Total Vehicles": [9220482, 8544913, 10049408, 9972399, 10704604, 10450961, 10733070, 10911615, 10221048, 10422293, 10083529, 10168032, 121482354],
        },
        "George Washington Bridge": {
            "Automobiles": [3515778, 3231728, 3800683, 3754742, 4027271, 3911105, 4075433, 4157650, 3831238, 3893123, 3834242, 3867206, 45900199],
            "Buses": [19873, 18800, 21365, 21234, 23022, 22515, 24343, 21754, 20174, 22091, 20551, 20481, 256203],
            "Trucks": [374286, 347881, 383736, 391817, 400208, 388808, 397526, 385734, 383394, 400534, 358187, 382908, 4595019],
            "Total Vehicles": [3909937, 3598409, 4205784, 4167793, 4450501, 4322428, 4497302, 4565138, 4234806, 4315748, 4212980, 4270595, 50751421],
        },
        "Lincoln Tunnel": {
            "Automobiles": [1110491, 1055381, 1274329, 1256524, 1404782, 1342522, 1317030, 1357493, 1303547, 1317680, 1268455, 1230106, 15238340],
            "Buses": [134254, 126148, 137773, 140474, 141048, 136217, 139201, 137429, 134168, 140480, 130729, 142578, 1640499],
            "Trucks": [85080, 80342, 90378, 91539, 92765, 90416, 95130, 91771, 98850, 105480, 87081, 90392, 1099224],
            "Total Vehicles": [1329825, 1261871, 1502480, 1488537, 1638595, 1569155, 1551361, 1586693, 1536565, 1563640, 1486265, 1463076, 17978063],
        },
        "Holland Tunnel": {
            "Automobiles": [1109523, 1041322, 1220569, 1201530, 1285769, 1265807, 1278796, 1313742, 1256902, 1287849, 1253668, 1282165, 14797642],
            "Buses": [3881, 3654, 4590, 4827, 5216, 5271, 5295, 4980, 3622, 4100, 4734, 4694, 54864],
            "Trucks": [34805, 32916, 36336, 37515, 39452, 38373, 39725, 38493, 39439, 40932, 35740, 39281, 453007],
            "Total Vehicles": [1148209, 1077892, 1261495, 1243872, 1330437, 1309451, 1323816, 1357215, 1299963, 1332881, 1294142, 1326140, 15305513],
        },
        "Goethals Bridge": {
            "Automobiles": [1268239, 1163060, 1379626, 1368917, 1450996, 1429138, 1469182, 1507810, 1374431, 1428837, 1370635, 1390507, 16601378],
            "Buses": [9116, 8474, 9186, 9154, 9620, 9567, 9561, 9257, 7812, 9154, 8322, 9236, 108459],
            "Trucks": [143404, 126665, 140975, 141785, 144926, 141702, 155499, 150325, 151219, 153951, 144051, 158638, 1753140],
            "Total Vehicles": [1420759, 1298199, 1529787, 1519856, 1605542, 1580407, 1634242, 1667392, 1533462, 1591942, 1523008, 1558381, 18462977],
        },
        "Outerbridge Crossing": {
            "Automobiles": [1058262, 987989, 1176907, 1167177, 1250175, 1252485, 1286142, 1297737, 1199990, 1199855, 1159577, 1150886, 14187182],
            "Buses": [1337, 1372, 1477, 1443, 1653, 1706, 1769, 1712, 1387, 1596, 1830, 1644, 18926],
            "Trucks": [53546, 47498, 56251, 56825, 56561, 54551, 59709, 56492, 57353, 57775, 53625, 54949, 665135],
            "Total Vehicles": [1113145, 1036859, 1234635, 1225445, 1308389, 1308742, 1347620, 1355941, 1258730, 1259226, 1215032, 1207479, 14871243],
        },
        "Bayonne Bridge": {
            "Automobiles": [278319, 253395, 295238, 306132, 349964, 340324, 356590, 357993, 336429, 337623, 331434, 318975, 3862416],
            "Buses": [1723, 1591, 1832, 1896, 1955, 1840, 1726, 1653, 1410, 1729, 1949, 1953, 21257],
            "Trucks": [18565, 16697, 18157, 18868, 19221, 18614, 20413, 19590, 19683, 19504, 18719, 21433, 229464],
            "Total Vehicles": [298607, 271683, 315227, 326896, 371140, 360778, 378729, 379236, 357522, 358856, 352102, 342361, 4113137],
        },
    },
    2026: {
        "All Crossings": {
            "Automobiles": [8042745, 7398232, None, None, None, None, None, None, None, None, None, None, 15440977],
            "Buses": [168879, 155418, None, None, None, None, None, None, None, None, None, None, 324297],
            "Trucks": [677541, 633462, None, None, None, None, None, None, None, None, None, None, 1311003],
            "Total Vehicles": [8889165, 8187112, None, None, None, None, None, None, None, None, None, None, 17076277],
        },
        "George Washington Bridge": {
            "Automobiles": [3387212, 3144863, None, None, None, None, None, None, None, None, None, None, 6532075],
            "Buses": [22169, 19977, None, None, None, None, None, None, None, None, None, None, 42146],
            "Trucks": [350382, 330174, None, None, None, None, None, None, None, None, None, None, 680556],
            "Total Vehicles": [3759763, 3495014, None, None, None, None, None, None, None, None, None, None, 7254777],
        },
        "Lincoln Tunnel": {
            "Automobiles": [1052240, 958955, None, None, None, None, None, None, None, None, None, None, 2011195],
            "Buses": [129739, 119863, None, None, None, None, None, None, None, None, None, None, 249602],
            "Trucks": [81158, 76764, None, None, None, None, None, None, None, None, None, None, 157922],
            "Total Vehicles": [1263137, 1155582, None, None, None, None, None, None, None, None, None, None, 2418719],
        },
        "Holland Tunnel": {
            "Automobiles": [1095942, 1015236, None, None, None, None, None, None, None, None, None, None, 2111178],
            "Buses": [4547, 4227, None, None, None, None, None, None, None, None, None, None, 8774],
            "Trucks": [35473, 33183, None, None, None, None, None, None, None, None, None, None, 68656],
            "Total Vehicles": [1135962, 1052646, None, None, None, None, None, None, None, None, None, None, 2188608],
        },
        "Goethals Bridge": {
            "Automobiles": [1227123, 1118491, None, None, None, None, None, None, None, None, None, None, 2345614],
            "Buses": [9117, 8252, None, None, None, None, None, None, None, None, None, None, 17369],
            "Trucks": [139500, 130847, None, None, None, None, None, None, None, None, None, None, 270347],
            "Total Vehicles": [1375740, 1257590, None, None, None, None, None, None, None, None, None, None, 2633330],
        },
        "Outerbridge Crossing": {
            "Automobiles": [1011483, 916141, None, None, None, None, None, None, None, None, None, None, 1927624],
            "Buses": [1573, 1499, None, None, None, None, None, None, None, None, None, None, 3072],
            "Trucks": [52530, 45495, None, None, None, None, None, None, None, None, None, None, 98025],
            "Total Vehicles": [1065586, 963135, None, None, None, None, None, None, None, None, None, None, 2028721],
        },
        "Bayonne Bridge": {
            "Automobiles": [268745, 244546, None, None, None, None, None, None, None, None, None, None, 513291],
            "Buses": [1734, 1600, None, None, None, None, None, None, None, None, None, None, 3334],
            "Trucks": [18498, 16999, None, None, None, None, None, None, None, None, None, None, 35497],
            "Total Vehicles": [288977, 263145, None, None, None, None, None, None, None, None, None, None, 552122],
        },
    },
}


def create_database(db_path="traffic_ezpass_usage.db"):
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
    CREATE TABLE traffic_monthly (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        year INTEGER NOT NULL,
        crossing TEXT NOT NULL,
        vehicle_type TEXT NOT NULL,
        month INTEGER NOT NULL CHECK (month BETWEEN 1 AND 12),
        month_name TEXT NOT NULL,
        traffic_count INTEGER,
        ezpass_usage_percent REAL,
        UNIQUE(year, crossing, vehicle_type, month)
    );

    CREATE TABLE traffic_ytd (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        year INTEGER NOT NULL,
        crossing TEXT NOT NULL,
        vehicle_type TEXT NOT NULL,
        traffic_count INTEGER,
        ezpass_usage_percent REAL,
        UNIQUE(year, crossing, vehicle_type)
    );
    """)

    monthly_rows = []
    ytd_rows = []

    for year, crossings in DATA.items():
        for crossing, metrics in crossings.items():
            for vehicle_type, values in metrics.items():
                is_ezpass = vehicle_type == "E-ZPass Usage (%)"
                ytd_value = values[-1]

                if is_ezpass:
                    vehicle = "All Vehicles"
                    for month_idx, value in enumerate(values[:12], start=1):
                        monthly_rows.append((year, crossing, vehicle, month_idx, MONTHS[month_idx - 1], None, value))
                    ytd_rows.append((year, crossing, vehicle, None, ytd_value))
                else:
                    for month_idx, value in enumerate(values[:12], start=1):
                        monthly_rows.append((year, crossing, vehicle_type, month_idx, MONTHS[month_idx - 1], value, None))
                    ytd_rows.append((year, crossing, vehicle_type, ytd_value, None))

    cur.executemany("""
        INSERT INTO traffic_monthly
            (year, crossing, vehicle_type, month, month_name, traffic_count, ezpass_usage_percent)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, monthly_rows)

    cur.executemany("""
        INSERT INTO traffic_ytd
            (year, crossing, vehicle_type, traffic_count, ezpass_usage_percent)
        VALUES (?, ?, ?, ?, ?)
    """, ytd_rows)

    conn.commit()
    conn.close()
    return db_path


if __name__ == "__main__":
    path = create_database()
    print(f"Created {path}")
