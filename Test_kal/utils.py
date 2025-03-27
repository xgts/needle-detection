import cv2


def plot_one_box(xywh,img, color=(0, 200, 0), target=False):
    xywh=[xywh[0],xywh[1],5,5]
    xyxy=xywh_to_xyxy(xywh)
    xy1 = (int(xyxy[0]), int(xyxy[1]))
    xy2 = (int(xyxy[2]), int(xyxy[3]))
    if target:
        color = (0, 0, 255)
    cv2.rectangle(img, xy1, xy2, color, 1, cv2.LINE_AA)

# if i%2==0:
            #     Z[0:4] = np.array([xywh]).T
            #     Z[4::] = np.array([dx, dy])
            # else:
            #     Z[0:4] = np.array([last_box_posterior]).T
            #     dx1 = last_box_posterior[0] - Z[0]
            #     dy1 = last_box_posterior[1] - Z[1]
            #     Z[4::] = np.array([dx1, dy1])
            # i+=1




def xywh_to_xyxy(xywh):
    x1 = xywh[0] - xywh[2] // 2
    y1 = xywh[1] - xywh[3] // 2
    x2 = xywh[0] + xywh[2] // 2
    y2 = xywh[1] + xywh[3] // 2

    return [x1, y1, x2, y2]


def updata_trace_list(box_center, trace_list, max_list_len=50):
    if len(trace_list) <= max_list_len:
        trace_list.append(box_center)
    else:
        trace_list.pop(0)
        trace_list.append(box_center)
    return trace_list

def draw_trace(img, trace_list):
    """
    更新trace_list,绘制trace
    :param trace_list:
    :param max_list_len:
    :return:
    """
    for i, item in enumerate(trace_list):

        if i < 1:
            continue
        cv2.line(img,
                 (trace_list[i][0], trace_list[i][1]), (trace_list[i - 1][0], trace_list[i - 1][1]),
                 (255, 255, 0), 3)
