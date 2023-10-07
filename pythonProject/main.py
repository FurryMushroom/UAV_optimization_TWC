# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import torch.autograd
from torch.distributions import Normal
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from _datetime import datetime as dt
from mpl_toolkits.mplot3d import Axes3D
NUM_OF_SAMPLE_POINTS = 5
sqrt_N_divided_by_2=torch.sqrt(torch.tensor(NUM_OF_SAMPLE_POINTS/2))
Nt = 5  # Nt denotes the number of isotropic antenna elements
β0_divide_square_σ=100000000  # square_σ denotes the power of AWGN at the receiver side,β0 represents the channel power gain at a reference distance of unit meter
theta = np.pi /12 # the angle in which antenna can gain radiation
false_alarm_rate=0.02
NUM_OF_UAVS=1
NUM_OF_TARGETS=3
NUM_OF_CHANNELS_OF_TARGETS=10
SAFE_REGION_OF_UAVS=1600
max_detection_frequencies_of_UAVs=torch.FloatTensor(NUM_OF_UAVS)
max_detection_frequencies_of_UAVs=torch.tensor([4000.])
# max_detection_frequencies_of_UAVs=torch.tensor([4000.,4000.,4000.])
# max_detection_frequencies_of_UAVs=torch.tensor([4000.,8000,12000,20000,28000,40000,60000,80000,120000,200000])
min_detection_frequencies_of_UAVs=torch.FloatTensor(NUM_OF_UAVS)
min_detection_frequencies_of_UAVs=torch.tensor([1000.])
# min_detection_frequencies_of_UAVs=torch.tensor([2000.,5000,7000,10000,18000,20000,34000,50000,60000,100000])
STANDARD_FREQUENCIES_OF_TARGETS=torch.FloatTensor(NUM_OF_TARGETS,1)
STANDARD_FREQUENCIES_OF_TARGETS=torch.tensor([[1000.],[1000.]])
# STANDARD_FREQUENCIES_OF_TARGETS=torch.tensor([[4000.],[9000],[16000],[60000],[180000]])  # initialize
CHANNEL_FREQUENCY_INTERVALS=torch.FloatTensor(NUM_OF_TARGETS,1)
CHANNEL_FREQUENCY_INTERVALS=torch.tensor([[300.],[300.]])
# CHANNEL_FREQUENCY_INTERVALS=torch.tensor([[1000.],[2000],[4000],[8000],[30000]])  # initialize
# TRANSMIT_POWER_OF_TARGETS=torch.FloatTensor(NUM_OF_TARGETS,1)
# TRANSMIT_POWER_OF_TARGETS=torch.tensor([.000004])
TRANSMIT_POWER_OF_TARGETS=.000004
M=Normal(torch.tensor(0.),torch.tensor(1.))
MIN_DISTANCE_BETWEEN_UAV_AND_TARGET=torch.tensor(500.)
MIN_DISTANCE_BETWEEN_UAVS=torch.tensor(50.)
q_xy=torch.FloatTensor(NUM_OF_UAVS,2) # position of UAVs
H=600. # height of UAVs
qt=torch.FloatTensor(NUM_OF_TARGETS,3)  # position of targets
q_xy=torch.tensor([[1530.,1690]
                   ])
q_xy.requires_grad=True
q_xy=q_xy*1
M1=torch.tensor([[1.,0,0],[0,1,0]])
M2=torch.tensor([0.,0.,H]).repeat(NUM_OF_UAVS,1)
qt=torch.tensor([[2900.,2900,500], # caution:location of UAVs at any dimension cannot be the same,will cause inf value and lead to error
                 [2100,2020,500],
                 [3300, 2300, 500],
                 # [2700, 1800, 500],
                 # [2100, 2300, 500],
                 # [2200, 3000, 500],
                 # [2600., 2600, 500],
                 # [2650, 3200, 500],
                 # [2400, 2100, 500],
                 # [2700, 1800, 500],
                 # [2000, 2850, 500],
                 # [2400, 2400, 500]
                 ])
azimuth=torch.FloatTensor(NUM_OF_UAVS,1) # denotes antenna azimuth of UAVs
azimuth=torch.tensor([[1.1]])
azimuth.requires_grad=True
azimuth=azimuth*1

with torch.autograd.set_detect_anomaly(True):
    glo_q_grad=torch.tensor((NUM_OF_UAVS,2),dtype=float)
    glo_azimuth_grad=torch.tensor((NUM_OF_UAVS,1),dtype=float)


    def extract_q(g):
        global glo_q_grad
        glo_q_grad = g


    def extract_azimuth(g):
        global glo_azimuth_grad
        glo_azimuth_grad = g


    def q_():
        q = torch.matmul(q_xy, M1) + M2
        return q


    def pos_relative():
        q=q_()
        q_unfolded=q.unfold(1,3,1)  # or the unfold function will be executed NUM_OF_TARGETS times,it seems
        position_relative=q_unfolded.repeat(1,NUM_OF_TARGETS,1).flatten(0,1)-qt.repeat(NUM_OF_UAVS,1)
        # unfold add one dimension,and repeat NUM_OF_TARGETS times at the new dimension,then flatten to 2 dims
        return position_relative


    def relative_hori_angle():
        position_relative=pos_relative()
        relative_horizontal_angle=torch.arctan(position_relative[:,1]/position_relative[:,0])
        return relative_horizontal_angle

    def angle_range():
        relative_angle=relative_hori_angle().reshape((NUM_OF_UAVS,NUM_OF_TARGETS))
        max_relative_angle=torch.max( relative_angle,dim=1)
        min_relative_angle = torch.min(relative_angle, dim=1)
        return min_relative_angle,max_relative_angle

    # def rela_pitch_angle():
    #     position_relative = pos_relative()
    #     relative_pitch_angle=torch.arctan(
    #         position_relative[:,2]/
    #         torch.sqrt(position_relative[:,1].pow(2)+position_relative[:,0].pow(2))
    #     )
    #     return relative_pitch_angle
    #
    #
    # def dist_between_UAVs_and_targets():
    #     position_relative=pos_relative()
    #     distance_between_UAVs_and_targets=torch.sqrt(position_relative[:,0].pow(2)+position_relative[:,1].pow(2)+position_relative[:,2].pow(2))
    #     distance_between_UAVs_and_targets=torch.reshape(distance_between_UAVs_and_targets,(9,1))
    #     # q_xy.register_hook(extract_q)
    #     # distance_between_UAVs_and_targets.backward(torch.ones(9, 1), retain_graph=True)
    #     return distance_between_UAVs_and_targets  # tensor(NUM_OF_TARGETS*NUM_OF_UAVs,)why?
    #
    #
    # def Directional_signal_gain():   # denotes directional radiation gain
    #     directional_signal_gain=torch.FloatTensor(NUM_OF_TARGETS*NUM_OF_UAVS,1)
    #     relative_horizontal_angle=relative_hori_angle()
    #     relative_pitch_angle=rela_pitch_angle()
    #     for i in range(0,NUM_OF_TARGETS*NUM_OF_UAVS):
    #         if abs(relative_horizontal_angle[i]-azimuth[i//NUM_OF_TARGETS]) < theta:
    #             directional_signal_gain[i]=torch.exp(
    #                 -((relative_horizontal_angle[i]-azimuth[i//NUM_OF_UAVS]).pow(2)+relative_pitch_angle[i].pow(2))/2/theta**2
    #             )
    #         else:
    #             directional_signal_gain[i]=0
    #     return directional_signal_gain

    def gamma():  # proportion between signal and noise
        # at first this was partitioned into several functions,but leads to repetitive calculation
        # position_relative
        q = q_()
        q_unfolded = q.unfold(1, 3, 1)  # or the unfold function will be executed NUM_OF_TARGETS times,it seems
        position_relative = q_unfolded.repeat(1, NUM_OF_TARGETS, 1).flatten(0, 1) - qt.repeat(NUM_OF_UAVS, 1)
        # unfold add one dimension,and repeat NUM_OF_TARGETS times at the new dimension,then flatten to 2 dims
        # relative_horizontal_angle
        relative_horizontal_angle = torch.arctan(position_relative[:, 1] / position_relative[:, 0])
        distance_between_UAVs_and_targets = torch.sqrt(
            position_relative[:, 0].pow(2) + position_relative[:, 1].pow(2) + position_relative[:, 2].pow(2))
        distance_between_UAVs_and_targets = torch.reshape(distance_between_UAVs_and_targets,
                                                          (NUM_OF_TARGETS * NUM_OF_UAVS, 1))

        directional_signal_gain = torch.FloatTensor(NUM_OF_TARGETS * NUM_OF_UAVS, 1)
        for i in range(0, NUM_OF_TARGETS * NUM_OF_UAVS):
            if abs(relative_horizontal_angle[i] - azimuth[i // NUM_OF_TARGETS]) < 0.95 * theta:  # caution:here the relative pitch angle was not taken into account
                directional_signal_gain[i] = torch.exp(
                    -((relative_horizontal_angle[i] - azimuth[i // NUM_OF_TARGETS]).pow(2)) / 2 / theta ** 2
                )
            elif abs(relative_horizontal_angle[i] - azimuth[i // NUM_OF_TARGETS]) < theta:
                directional_signal_gain[i] = torch.exp(-torch.tensor(0.95 ** 2 / 2)) * 20 * (
                            1 - 1 * abs(relative_horizontal_angle[i] - azimuth[i // NUM_OF_TARGETS]) / theta)
            else:
                directional_signal_gain[i] = 0
        gamma_ = TRANSMIT_POWER_OF_TARGETS * directional_signal_gain * β0_divide_square_σ * Nt / distance_between_UAVs_and_targets
        return gamma_


    def inversed_right_tailed_function(x):
        return M.icdf(1-x)  # x is the false alarm rate


    inversed_right_tailed_false_alarm_rate=inversed_right_tailed_function(torch.tensor(false_alarm_rate))  # a const in overall algorithm


    def right_tailed_function(x):
        return 1-M.cdf(x)


    def energy_detection_probility():
        gamma_=gamma()
        # q_xy.register_hook(extract_q)
        # gamma_.backward(torch.ones(9, 1), retain_graph=True)
        energy_detection_probility_= right_tailed_function(
            (inversed_right_tailed_false_alarm_rate.repeat(NUM_OF_TARGETS*NUM_OF_UAVS,1)-sqrt_N_divided_by_2*gamma_) /
            (1+gamma_)
        )
        # q_xy.register_hook(extract_q)
        # energy_detection_probility_.backward(torch.ones(9, 1),retain_graph=True)
        return energy_detection_probility_



    def P_q_psai():
        energy_detection_probility_= energy_detection_probility()
        P_q_psai=energy_detection_probility_.cumsum(dim=0)[NUM_OF_UAVS*NUM_OF_TARGETS-1]
        return P_q_psai

    def plot_fig():
        fig = plt.figure()  # 定义画布
        x_major_locator = MultipleLocator(500)  # 把x轴的刻度间隔设置，并存在变量里
        y_major_locator = MultipleLocator(500)
        ax = plt.gca()  # ax为两条坐标轴的实例
        ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置
        ax.yaxis.set_major_locator(y_major_locator)
        plt.xlim(0, 4500)  # 把x轴的刻度范围设置，不满一个刻度间隔数字不会显示出来，但是能看到一点空白
        plt.ylim(0, 4500)
        LENGTH = 3000
        x = np.array([i for i in range(0, 1600)])
        plt.fill_between(x, 0, 4500, facecolor='green', alpha=0.2)

        def plot_edge_ray(start_x, start_y, angle):  # plot a pseudo ray
            for i in range(0, start_x.size):
                end_x = start_x[i] + LENGTH * np.cos(angle[i])
                end_y = start_y[i] + LENGTH * np.sin(angle[i])
                ray_x = np.append(start_x[i], end_x)
                ray_y = np.append(start_y[i], end_y)
                plt.plot(ray_x, ray_y, "g:", linewidth=.5)

        def plot_axis_ray(start_x, start_y, angle):  # plot a pseudo ray
            for i in range(0, start_x.size):
                print(start_x[i])
                end_x = start_x[i] + LENGTH * np.cos(angle[i])
                end_y = start_y[i] + LENGTH * np.sin(angle[i])
                ray_x = np.append(start_x[i], end_x)
                ray_y = np.append(start_y[i], end_y)
                plt.plot(ray_x, ray_y, color="dodgerblue", linestyle='--', linewidth=.6)

        q_plot = q_xy.detach().numpy()
        azi = azimuth.detach().numpy()
        plt.scatter(q_plot[:, 0], q_plot[:, 1], c="b", marker='*')
        plt.scatter(qt[:, 0], qt[:, 1], c="r", marker='x')
        plot_axis_ray(q_plot[:, 0], q_plot[:, 1], azi)
        plot_edge_ray(q_plot[:, 0], q_plot[:, 1], azi - theta)
        plot_edge_ray(q_plot[:, 0], q_plot[:, 1], azi + theta)
        for i in range(0, NUM_OF_TARGETS):
            draw_circle = plt.Circle((qt[i][0], qt[i][1]), MIN_DISTANCE_BETWEEN_UAV_AND_TARGET, color='lightsalmon',
                                     fill=False, ls='--', linewidth=0.4)
            plt.gcf().gca().add_artist(draw_circle)
        ax.set_aspect(1)  # set the axes proportion
        timestamp = dt.strftime(dt.now(), '%Y_%m_%d_%Hh%Mm%Ss')
        plt.savefig('figs/fig_' + timestamp + '.png', dpi=1000)
        plt.show()


    χ=torch.zeros((NUM_OF_TARGETS*NUM_OF_UAVS,1),dtype=torch.float32)  # Lagrange multiplier
    b=torch.zeros((NUM_OF_TARGETS*NUM_OF_UAVS,3),dtype=torch.float32)
    µ=torch.zeros((NUM_OF_UAVS*(NUM_OF_UAVS-1)//2,1) ,dtype=torch.float32)  # Lagrange multiplier
    c=torch.zeros((NUM_OF_UAVS*(NUM_OF_UAVS-1)//2,3) ,dtype=torch.float32)

    omiga_b,omiga_c=2,2
    epsilon_1b,epsilon_1c,epsilon_2b,epsilon_2c=.95,.95,1.05,1.05
    A2=torch.FloatTensor(NUM_OF_UAVS*(NUM_OF_UAVS-1)//2,NUM_OF_UAVS)
    ita=torch.tensor(0.000001)  # iteration threshold
    rows=0
    for i in range(0,NUM_OF_UAVS-1):
        for j in range(i+1,NUM_OF_UAVS):
            A2[rows][j]=-1
            A2[rows][i] = 1
            rows=rows+1


    def delta_rb():
        q=q_()
        return torch.norm(q.unfold(1, 3, 1).repeat(1,NUM_OF_TARGETS, 1).flatten(0, 1) - qt.repeat(NUM_OF_UAVS, 1)-b)  # definition of norm is questionable


    def delta_rc():
        q=q_()
        return torch.norm(torch.matmul(A2,q)-c)


    def absolute_of_biggest_element(X):
        length=X.shape[0]
        max_abs=0
        for i in range(0,length):
            t=torch.norm(X[i])
            if t>max_abs:
                max_abs=t
        return max_abs

    def P_adpm():
        P_q_psai_=P_q_psai()
        # q_xy.register_hook(extract_q)
        # P_q_psai_.backward(retain_graph=True)
        return -P_q_psai_+ρ2/2*torch.norm(torch.matmul(A2,q)-c+µ).pow(2)-ρ2/2*torch.norm(µ).pow(2) +\
                ρ1/2* torch.norm(q.unfold(1, 3, 1).repeat(1,NUM_OF_TARGETS, 1).flatten(0, 1) - qt.repeat(NUM_OF_UAVS, 1)-b+χ).pow(2)-\
                ρ1 / 2 * torch.norm( χ).pow(2)

    MAX_ITERATION_TIMES = 200
    MAX_ITERATION_TIMES_Q = 50
    MAX_ITERATION_TIMES_AZI = 5
    NUM_OF_SEARCH_DIRECTIONS=3
    original_alpha_q ,original_alpha_azi = 800,.01
    threshold_q,threshold_azi = .00000001,0.001
    ρ1, ρ2 = .00002, .00002  # penalty factor

    # begin iteration
    iter_times=0
    while 1:
        if iter_times< MAX_ITERATION_TIMES//2*1:
            alpha_q, alpha_azi =  original_alpha_q, original_alpha_azi
        elif  iter_times< MAX_ITERATION_TIMES//5*4:
            alpha_q, alpha_azi = original_alpha_q / 4, original_alpha_azi / 4
        else:
            alpha_q, alpha_azi = original_alpha_q / 16, original_alpha_azi / 16

        delta_rb_before=delta_rb()
        delta_rc_before=delta_rc()
        q=q_()
        # update b
        for i in range(0,NUM_OF_UAVS):
            for j in range(0, NUM_OF_TARGETS ):
                v=i*NUM_OF_TARGETS+j
                expression1=q[i]-qt[j]+χ[v]
                ξ1=torch.norm(expression1)
                b[v]=expression1*torch.max( ξ1,MIN_DISTANCE_BETWEEN_UAV_AND_TARGET)/ ξ1
        # update c
        v=0
        for i in range(0,NUM_OF_UAVS-1):
            for j in range(i+1, NUM_OF_UAVS ):
                expression2=q[i]-q[j]+µ[v]
                ξ2=torch.norm(expression2)
                c[v]=expression2*torch.max( ξ2,MIN_DISTANCE_BETWEEN_UAVS)/ ξ2
                v = v + 1

        delta_rb_after = delta_rb()
        delta_rc_after = delta_rc()
        # update q
        iter_times_q=0
        while True:
            target_func= P_adpm()
            q_xy.register_hook(extract_q)
            target_func.backward(retain_graph=True)
            # q_ba=q_xy-glo_q_grad
            # for i in range(0,NUM_OF_UAVS):
            #     if q_ba[i][0]>SAFE_REGION_OF_UAVS:
            #         q_ba[i][0]=SAFE_REGION_OF_UAVS
            # q_xy=q_xy+alpha_1*(q_ba-q_xy)
            q_xy = q_xy - alpha_q*glo_q_grad
            for i in range(0, NUM_OF_UAVS):
              if q_xy[i][0]>SAFE_REGION_OF_UAVS:
                    q_xy[i][0]=SAFE_REGION_OF_UAVS
            iter_times_q=iter_times_q+1
            if torch.norm(glo_q_grad)<threshold_q or iter_times_q>=MAX_ITERATION_TIMES_Q:
                break
        q=q_()
        # update azimuth
        iter_times_azimuth = 0
        # add random search,dispense dependency on initial value
        min_hori_angle,max_hori_angle=angle_range()
        for i in range(0, NUM_OF_UAVS):
            for j in range(0, NUM_OF_SEARCH_DIRECTIONS):
                azimuth_pre = azimuth[i].clone()
                target_func_pre=P_q_psai()
                azimuth[i] =((NUM_OF_SEARCH_DIRECTIONS-j-1)*min_hori_angle[0][i]+j*max_hori_angle[0][i]) \
                            /(NUM_OF_SEARCH_DIRECTIONS-1)
                # between min_hori_angle and max_hori_angle=angle_range(),get average dispersed directions
                target_func=P_q_psai()
                if(target_func>target_func_pre):
                    pass
                else:
                    azimuth[i]=azimuth_pre
        # grad descent
        while 1:
            target_func = -P_q_psai()
            azimuth.register_hook(extract_azimuth)
            target_func.backward(retain_graph=True)
            azimuth=azimuth-alpha_azi* glo_azimuth_grad
            iter_times_azimuth = iter_times_azimuth + 1
            if torch.norm( glo_azimuth_grad)<threshold_azi or iter_times_azimuth>MAX_ITERATION_TIMES_AZI:
                break
        # project into -pi to pi
        for i in range(0, NUM_OF_UAVS):
            while True:
                if azimuth[i] > np.pi:
                    azimuth[i] = azimuth[i] - 2 * np.pi
                elif azimuth[i] < -np.pi:
                    azimuth[i] = azimuth[i] + 2 * np.pi
                else:
                    break
        #update ρ1,ρ2
        ρ1_before, ρ2_before=ρ1,ρ2
        if delta_rb_after>delta_rb_before*epsilon_1b:
            ρ1=ρ1*epsilon_2b
        if delta_rc_after>delta_rc_before*epsilon_1c:
            ρ2=ρ2*epsilon_2c
        # update µ,χ
        µ=(µ*ρ2/ρ2_before+torch.matmul(A2,q)-c).detach()
        χ= (χ*ρ1/ρ1_before+q.unfold(1, 3, 1).repeat(1,NUM_OF_TARGETS, 1).flatten(0, 1) - qt.repeat(NUM_OF_UAVS, 1)-b).detach()
        absolute_of_biggest_element_of_χ=absolute_of_biggest_element(χ)
        absolute_of_biggest_element_of_µ = absolute_of_biggest_element(µ)
        if absolute_of_biggest_element_of_χ>omiga_b:
            χ=χ/absolute_of_biggest_element_of_χ
        if absolute_of_biggest_element_of_µ > omiga_c:
            µ =µ / absolute_of_biggest_element_of_µ
        #  whether or not have converged

        iter_times=iter_times+1
        # if delta_rc_before+delta_rb_before<ita or iter_times > MAX_ITERATION_TIMES:
        #     break
        if iter_times > MAX_ITERATION_TIMES:
            plot_fig()
            print(q_xy)
            print(azimuth)

            break
# # plot
# fig = plt.figure()  # 定义画布
# xs=q[:,0].detach().numpy()
# ys=q[:,1].detach().numpy()
# x_ts=qt[:,0].detach().numpy()
# y_ts=qt[:,1].detach().numpy()
# azi_x=q_xy[0].numpy()
# azi_y=q_xy[1].numpy()
# azi_x.append(azi_x+1000*azi_x*np.cos(azimuth.numpy()))
# azi_y.append(azi_y+1000*azi_y*np.sin(azimuth.numpy()))
# plt.scatter(xs, ys)
# plt.scatter(x_ts,y_ts)
# plt.plot(azi_x, azi_y,"r--")
# plt.show()
#
#
# zs=np.ones(NUM_OF_UAVS)*H
# ax = plt.axes(projection='3d')  # 定义三维坐标轴 或者：ax2 = Axes3D(fig)
# ax.scatter3D(xs,ys,zs, c='b')
#
#
#
#
#
#
#
#
#














