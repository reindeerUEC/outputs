# jkobo_main.py  by KUDOH Shunsuke
# --- main program for Joho-kogaku kobo

# 大学の授業で作成した、ロボットの自律走行プログラムです。
# 先生のコードを元に大幅に書き直して、同じクラスを履修した人との間の対戦で1位になりました

# ロボットはカメラで自分とボールとゴールの位置を認識し、自律走行します。
# ボールがゴールに入ると終了します。

import math
import sys
import time

import cv2
import jkobo_robot as jkr
import jkobo_vision as vis
import numpy as np

global position_dict

position_dict = {
    "robo_pos": None,
    "oppo_robo_pos": None,
    "ball_pos": None,
    "goal_pos": None,
    "oppo_goal_pos": None,
    "robo_dir": None,
    "oppo_robo_dir": None,
    "goal_dir": None,
    "oppo_goal_dir": None,
    "robo_pos_Null": "False",
    "oppo_robo_pos_Null": "False",
    "ball_pos_Null": "False",
    "goal_pos_Null": "False",
    "oppo_goal_pos_Null": "False",
    "robo_dir_Null": "False",
    "oppo_robo_dir_Null": "False",
    "goal_dir_Null": "False",
    "ball_uv": None,
    "ball_uv_Null": "False",
    "img": None,
    "img_small": None,
}


def position():
    global position_dict
    i = 0
    while i < 4:
        _, img = cap.read()
        i += 1

    # ball detection
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ball_pos, ball_uv = vis.find_ball(
        img_hsv, ball_lowerb, ball_upperb, ball_h, rvec_w, tvec_w, mtx, dst
    )

    # robot detection
    robo_pos, robo_dir, rvec_r, tvec_r = vis.find_robot(
        img, robo_id, aruco_len, aruco_dictionary, rvec_w, tvec_w, mtx, dst
    )

    # oppo_robot detection
    oppo_robo_pos, oppo_robo_dir, rvec_or, tvec_or = vis.find_robot(
        img, oppo_robo_id, aruco_len, aruco_dictionary, rvec_w, tvec_w, mtx, dst
    )

    # goal detection
    goal_pos, goal_dir, rvec_g, tvec_g = vis.find_robot(
        img, goal_id, aruco_len, aruco_dictionary, rvec_w, tvec_w, mtx, dst
    )

    # oppo_goal detection
    oppo_goal_pos, oppo_goal_dir, rvec_og, tvec_og = vis.find_robot(
        img, oppo_goal_id, aruco_len, aruco_dictionary, rvec_w, tvec_w, mtx, dst
    )

    img_small = infoOnimg(
        img,
        ball_pos,
        robo_pos,
        goal_pos,
        robo_dir,
        goal_dir,
        oppo_robo_pos,
        oppo_goal_pos,
        oppo_robo_dir,
        oppo_goal_dir,
        ball_uv,
        rvec_r,
        tvec_r,
        rvec_g,
        tvec_g,
        rvec_w,
        tvec_w,
        rvec_or,
        tvec_or,
        rvec_og,
        tvec_og,
    )

    cv2.imshow("jkobo_main", img_small)
    key = cv2.waitKey(1)
    if key == 113:
        sys.exit(1)
        # write code to jump to the if___name__ == "__main__" part
        # return

    renew_position_dict(
        robo_pos,
        ball_pos,
        goal_pos,
        robo_dir,
        goal_dir,
        oppo_robo_pos,
        oppo_goal_pos,
        oppo_robo_dir,
        oppo_goal_dir,
        ball_uv,
        img,
        img_small,
    )


def infoOnimg(
    img,
    ball_pos,
    robo_pos,
    goal_pos,
    robo_dir,
    goal_dir,
    oppo_robo_pos,
    oppo_goal_pos,
    oppo_robo_dir,
    oppo_goal_dir,
    ball_uv,
    rvec_r,
    tvec_r,
    rvec_g,
    tvec_g,
    rvec_w,
    tvec_w,
    rvec_or,
    tvec_or,
    rvec_og,
    tvec_og,
):
    if ball_pos is not None:
        ball_str = "Ball:  [{:>6.1f} {:>6.1f}]".format(ball_pos[0], ball_pos[1])
        cv2.drawMarker(img, ball_uv, (0, 255, 255), thickness=3, markerSize=40)
    else:
        ball_str = "Ball: None"

    if robo_pos is not None:
        robo_str = (
            "Robot({}):".format(robo_id)
            + " pos[{:>6.1f} {:>6.1f}]".format(robo_pos[0], robo_pos[1])
            + " dir[{:>6.1f} {:>6.1f}]".format(robo_dir[0], robo_dir[1])
        )
        cv2.drawFrameAxes(img, mtx, dst, rvec_r, tvec_r, 25)
    else:
        robo_str = "Robot: None"

    if goal_pos is not None:
        goal_str = (
            "Goal({}):".format(goal_id)
            + " pos[{:>6.1f} {:>6.1f}]".format(goal_pos[0], goal_pos[1])
            + " dir[{:>6.1f} {:>6.1f}]".format(goal_dir[0], goal_dir[1])
        )
        cv2.drawFrameAxes(img, mtx, dst, rvec_g, tvec_g, 25)
    else:
        goal_str = "Goal: None"

    if oppo_robo_pos is not None:
        oppo_robo_str = (
            "Opponent Robot({}):".format(oppo_robo_id)
            + " pos[{:>6.1f} {:>6.1f}]".format(oppo_robo_pos[0], oppo_robo_pos[1])
            + " dir[{:>6.1f} {:>6.1f}]".format(oppo_robo_dir[0], oppo_robo_dir[1])
        )
        cv2.drawFrameAxes(img, mtx, dst, rvec_or, tvec_or, 25)
    else:
        oppo_robo_str = "Opponent Robot: None"

    if oppo_goal_pos is not None:
        oppo_goal_str = (
            "Opponent Goal({}):".format(oppo_goal_id)
            + " pos[{:>6.1f} {:>6.1f}]".format(oppo_goal_pos[0], oppo_goal_pos[1])
            + " dir[{:>6.1f} {:>6.1f}]".format(oppo_goal_dir[0], oppo_goal_dir[1])
        )
        cv2.drawFrameAxes(img, mtx, dst, rvec_og, tvec_og, 25)
    else:
        oppo_goal_str = "Opponent Goal: None"

    cv2.drawFrameAxes(img, mtx, dst, rvec_w, tvec_w, 25)

    # write information and show the image
    img_small = img if ss == 1 else cv2.resize(img, None, fx=ss, fy=ss)
    cv2.putText(
        img_small, ball_str, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2
    )
    cv2.putText(
        img_small, robo_str, (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2
    )
    cv2.putText(
        img_small, goal_str, (10, 90), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2
    )
    cv2.putText(
        img_small, oppo_robo_str, (10, 120), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2
    )
    cv2.putText(
        img_small, oppo_goal_str, (10, 150), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2
    )
    return img_small


def renew_position_dict(*args, ball_uv=None, img=None, img_small=None):
    global position_dict

    key_value_pairs = {
        "robo_pos": args[0],
        "ball_pos": args[1],
        "goal_pos": args[2],
        "robo_dir": args[3],
        "goal_dir": args[4],
        "oppo_robo_pos": args[5],
        "oppo_goal_pos": args[6],
        "oppo_robo_dir": args[7],
        "oppo_goal_dir": args[8],
        "ball_uv": ball_uv,
        "img": img,
        "img_small": img_small,
    }

    for key, value in key_value_pairs.items():
        null_key = f"{key}_Null"
        position_dict[null_key] = "False" if value is not None else "True"
        if value is not None:
            position_dict[key] = value


def calc_distance_and_neighbor_point(a, b, p):
    """
    線分abと点pとの距離
    """
    ap = p - a
    ab = b - a
    ba = a - b
    bp = p - b
    if np.dot(ap, ab) < 0:
        distance = np.linalg.norm(ap)
        neighbor_point = a
    elif np.dot(bp, ba) < 0:
        distance = np.linalg.norm(bp)
        neighbor_point = b
    else:
        ai_norm = np.dot(ap, ab) / np.linalg.norm(ab)
        neighbor_point = a + (ab) / np.linalg.norm(ab) * ai_norm
        distance = np.linalg.norm(p - neighbor_point)
    return (neighbor_point, distance)


def judgecollusion(robo_pos, ball_pos, purpose_pos):
    """
    robotの通り道にballがぶつかるか判定
    """
    _, distance = calc_distance_and_neighbor_point(purpose_pos, robo_pos, ball_pos)
    if distance < (robo_radius + ball_radius) * 1.2:
        return True
    else:
        return False


def th2dir(th):
    """convert angle[rad] to direction vector
    Args:
        th : angle[rad] with x-axis
    Returns:
        ndarray : direction vector ([dx, dy])
    """
    return np.array([math.cos(th), math.sin(th)])


def dir2th(dir):
    """convert direction vector to angle[rad]
    Args:
        dir : direction vector ([dx, dy])
    Returns:
        float : angle[rad] with x-axis
    """
    return math.atan2(dir[1], dir[0])


def avoidCollision(robo_pos, ball_pos, purpose_pos):
    """
    回り道を求めて、経由地を返す
    近い方をavoid_posにする
    """
    x = np.deg2rad(90)
    rot = np.array([[math.cos(x), -math.sin(x)], [math.sin(x), math.cos(x)]])
    avoid_pos1 = ball_pos + np.dot(rot, purpose_pos - ball_pos)
    y = np.deg2rad(-90)
    rot = np.array([[math.cos(y), -math.sin(y)], [math.sin(y), math.cos(y)]])
    avoid_pos2 = ball_pos + np.dot(rot, purpose_pos - ball_pos)
    if np.linalg.norm(robo_pos - avoid_pos1) > np.linalg.norm(robo_pos - avoid_pos2):
        return avoid_pos2
    else:
        return avoid_pos1


def overGoal(ball_pos, goal_pos, goal_dir):
    """
    ボールがゴールに入っているか判定
    引数:ボールとゴールの場所、ゴールのベクトル(両方ndarray)
    計算:ボールとゴールのベクトルの外積を用いて角度。
    返り値:True or False
    """
    ball_dir = ball_pos - goal_pos  # ゴールよりちょい後ろをゴールラインにする
    dot = np.dot(ball_dir, goal_dir)
    cos = dot / (np.linalg.norm(ball_dir) * np.linalg.norm(goal_dir))  # cosを計算
    sin = math.sqrt(1 - cos**2)  # sinを計算
    length = (
        np.linalg.norm(ball_pos - goal_pos) * sin
    )  # ボールとゴールの距離(ゴールライン上での正射影)

    # ゴールは全長700mmなので、ボールがゴールの外側にあるか判定
    if cos < 0 and length < 355:
        return True
    else:
        return False


def culcTurnRadius(robo_pos, purpose_pos, robo_dir):
    radius = math.atan2(
        np.cross(robo_dir, purpose_pos - robo_pos),
        np.dot(robo_dir, purpose_pos - robo_pos),
    )
    return radius


def ifGoStrait(robo_pos, ball_pos, goal_pos):
    """
    ロボットとゴールの間にボールがあるか判定
    引数：ロボット、ボール、ゴールの場所
    計算：ロボットとゴール間の線分とボールとの距離を計算し、まっすぐ行っていいか判定
    返り値:True or False
    """
    _, distance = calc_distance_and_neighbor_point(robo_pos, goal_pos, ball_pos)
    if distance < robo_radius / 3:  # ここは要調整
        return True
    else:
        return False


def find_intersection(point1, vector1, point2, vector2):
    """
    二つの線分を構成する点とベクトルを受け取り、交点を返す
    """
    if point1 is None or point2 is None or vector1 is None or vector2 is None:
        return 0
    x1, y1 = point1
    x2, y2 = point2
    u1, v1 = vector1
    u2, v2 = vector2

    # 例外処理: 二つのベクトルが平行な場合
    if u1 * v2 - u2 * v1 == 0:
        # raise ValueError("ベクトルが平行です。交点が存在しません。")
        return 0

    # 交点の座標を計算
    t = ((x2 - x1) * v2 - (y2 - y1) * u2) / (u1 * v2 - u2 * v1)
    x_intersection = x1 + t * u1
    y_intersection = y1 + t * v1

    distance = np.linalg.norm(
        np.array([x_intersection, y_intersection]) - np.array([x1, y1])
    )
    return distance


def goStraight(self):
    print("go straight")
    purpose_pos = position_dict["goal_pos"]
    turn_radius = culcTurnRadius(
        position_dict["robo_pos"], purpose_pos, position_dict["robo_dir"]
    )
    self.highspeedturn(turn_radius)
    self.highspeedmove(np.linalg.norm(position_dict["robo_pos"] - purpose_pos))
    self.highspeedmove(-np.linalg.norm(position_dict["robo_pos"] - purpose_pos) / 2)
    print("end straight")


def goCurve(self):
    print("go curve")
    purpose_pos = position_dict["ball_pos"] - (
        position_dict["goal_pos"] - position_dict["ball_pos"]
    ) / np.linalg.norm(position_dict["goal_pos"] - position_dict["ball_pos"]) * 3 * (
        robo_radius + ball_radius
    )
    turn_radius = culcTurnRadius(
        position_dict["robo_pos"], purpose_pos, position_dict["robo_dir"]
    )
    self.highspeedturn(turn_radius)
    self.highspeedmove(np.linalg.norm(position_dict["robo_pos"] - purpose_pos))
    ball_goal_vector = position_dict["goal_pos"] - position_dict["ball_pos"]
    distance = find_intersection(
        position_dict["robo_pos"],
        position_dict["robo_dir"],
        position_dict["ball_pos"],
        ball_goal_vector,
    )

    # if distance < 100:
    #     self.highspeedmove(-distance)
    print("end curve")


def avoid(self):
    print("avoid")
    purpose_pos = position_dict["ball_pos"] - (
        position_dict["goal_pos"] - position_dict["ball_pos"]
    ) / np.linalg.norm(position_dict["goal_pos"] - position_dict["ball_pos"]) * 3 * (
        robo_radius + ball_radius
    )
    avoid_pos = avoidCollision(
        position_dict["robo_pos"],
        position_dict["ball_pos"],
        purpose_pos,
    )
    purpose_pos = avoid_pos
    self.highspeedturn(
        culcTurnRadius(
            position_dict["robo_pos"], purpose_pos, position_dict["robo_dir"]
        )
    )
    self.highspeedmove(np.linalg.norm(position_dict["robo_pos"] - purpose_pos))
    print("end avoid")


def run_robot(self):
    global position_dict

    # position()

    if (
        position_dict["robo_pos"] is None
        or position_dict["ball_pos"] is None
        or position_dict["goal_pos"] is None
        or position_dict["robo_dir"] is None
        or position_dict["goal_dir"] is None
        or position_dict["oppo_robo_pos"] is None
        or position_dict["oppo_goal_pos"] is None
        or position_dict["oppo_robo_dir"] is None
        or position_dict["oppo_goal_dir"] is None
    ):
        print("error: cannot read positions correctly")
        sys.exit(-1)

    start_time = time.time()
    # first_flag = True
    # if first_flag:  # 初回のみ
    #     first_flag = False
    #     self.highspeedmove(
    #         np.linalg.norm(position_dict["robo_pos"] - position_dict["goal_pos"])
    #     )
    #     self.highspeedmove(
    #         -np.linalg.norm(position_dict["robo_pos"] - position_dict["goal_pos"]) / 3
    #     )
    #     position()

    while True:
        purpose_pos = position_dict["goal_pos"]
        if not judgecollusion(
            purpose_pos, position_dict["robo_pos"], position_dict["ball_pos"]
        ):
            if ifGoStrait(
                position_dict["robo_pos"],
                position_dict["ball_pos"],
                position_dict["goal_pos"],
            ):
                goStraight(self)
            else:
                goCurve(self)

        else:
            avoid(self)
        position()
        backword = np.linalg.norm(
            position_dict["robo_pos"] - position_dict["goal_pos"]
        )  # ロボットが見えないときに戻る距離
        if position_dict["robo_pos_Null"] == "True":
            print("robo_pos is None")
            self.highspeedmove(-backword / 2)
            position()

        # if overGoal(
        #     position_dict["ball_pos"],
        #     position_dict["goal_pos"],
        #     position_dict["goal_dir"],
        # ):
        #     print("goal")
        #     break
        # if overGoal(
        #     position_dict["ball_pos"],
        #     position_dict["oppo_goal_pos"],
        #     position_dict["oppo_goal_dir"],
        # ):
        #     print("oppo_goal")
        #     break
        if time.time() - start_time > battletime:  # 30秒たったら終了
            print("timeout")
            break
    return


def run_robot2(self):
    global position_dict
    # position()

    if (
        position_dict["robo_pos"] is None
        or position_dict["ball_pos"] is None
        or position_dict["goal_pos"] is None
        or position_dict["robo_dir"] is None
        or position_dict["goal_dir"] is None
        or position_dict["oppo_robo_pos"] is None
        or position_dict["oppo_goal_pos"] is None
        or position_dict["oppo_robo_dir"] is None
        or position_dict["oppo_goal_dir"] is None
    ):
        print("error: cannot read positions correctly")
        sys.exit(-1)

    start_time = time.time()

    while True:
        purpose_pos = position_dict["goal_pos"]
        if not judgecollusion(
            purpose_pos, position_dict["robo_pos"], position_dict["ball_pos"]
        ):
            if ifGoStrait(
                position_dict["robo_pos"],
                position_dict["ball_pos"],
                position_dict["goal_pos"],
            ):
                goStraight(self)
            else:
                goCurve(self)

        else:
            avoid(self)
        position()
        backword = np.linalg.norm(
            position_dict["robo_pos"] - position_dict["goal_pos"]
        )  # ロボットが見えないときに戻る距離
        if position_dict["robo_pos_Null"] == "True":
            print("robo_pos is None")
            self.highspeedmove(-backword / 2)
            position()

        if time.time() - start_time > battletime:  # 30秒たったら終了
            print("timeout")
            break
    return


if __name__ == "__main__":
    # parameter setting
    robot = jkr.JkoboRobot(ip_address="192.168.4.161")
    calib_file = "calib_C3.json"  # camera calibration file
    rth_file = "rt.json"  # RT file (the world frame)
    robo_id = 1  # marker ID of robot
    oppo_robo_id = 3  # marker ID of opponent
    goal_id = 10  # marker ID of goal
    oppo_goal_id = 0  # marker ID of opponent's goal

    # read camera parameters
    mtx, dst, _ = vis.read_calib_file(calib_file)
    if mtx is None or dst is None:
        print("error: cannot read camera parameters correctly: " + calib_file)
        sys.exit(-1)

    # read an RT file
    rvec_w, tvec_w = vis.read_RT_file(rth_file)
    if rvec_w is None:
        print("error: cannot read the world frame correctly: " + rth_file)
        sys.exit(-1)

    # fixed parameters
    robo_radius = 60
    ball_radius = 20
    term = 250  # 半カーペット用に合わせた値
    ball_h = 20  # height of the ball center [mm]
    ball_lowerb = np.array([170, 200, 100])  # lower boundary of ball color HSV
    ball_upperb = np.array([190, 255, 255])  # upper boundary of ball color
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_len = 80  # length of ArUco marker's side [mm]

    # open camera
    # cap = cv2.VideoCapture(camera_no)
    # for Windows users (as needed)
    battletime = 1000  # 試合時間[s]
    camera_no = 0  # camera device no. 0か1
    ss = 0.5  # scale factor for displaying images
    cap = cv2.VideoCapture(camera_no, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        position()

        print("waiting")
        key = cv2.waitKey(1)
        if key == 113:  # q
            break

        # elif key == 99:  # c
        #     print("enter")
        #     run_robot2(robot)

        elif key == 13:  # Enter
            print("enter")
            run_robot(robot)

# EOF
