import torch
import numpy as np 

def angle2cord(angles, device=torch.device('cpu')):
    # angles are B x 5 x 4
    B = angles.shape[0]
    pi = torch.tensor(np.pi).to(device)

    # define the parameters
    phi_I = 110 * pi / 180
    phi_M = 90 * pi / 180
    phi_R = 70 * pi / 180
    phi_L = 50 * pi / 180

    l = torch.zeros(20,20).to(device)
    l[0][1] = 7.0
    l[0][2] = 9.0
    l[0][3] = 8.5
    l[0][4] = 8.0
    l[0][5] = 8.0
    l[1][6] = 3.5
    l[6][7] = 3.0
    l[2][8] = 4.5
    l[8][9] = 2.5
    l[9][10] = 2.0
    l[3][11] = 5.0
    l[11][12] = 3.0
    l[12][13] = 2.0
    l[4][14] = 4.5
    l[14][15] = 3.0
    l[15][16] = 2.0
    l[5][17] = 4.0
    l[17][18] = 2.5
    l[18][19] = 2.0

    ### input ###
    phi = torch.zeros(B, 6).to(device)
    theta = torch.zeros(B, 20).to(device)

    phi[:,[0,2,3,4,5]] = angles[:,:,0]
    theta[:,[0,2,3,4,5]] = angles[:,:,1]
    theta[:,[1,8,11,14,17]] = angles[:,:,2]
    theta[:,[6,9,12,15,18]] = angles[:,:,3]

    e = torch.zeros(B,20,3).to(device)
    # four digit PIPs
    e[:,2] = e[:,0] + l[0][2] * torch.tensor([[torch.cos(phi_I), torch.sin(phi_I), 0]]).to(device).repeat(B,1)
    e[:,3] = e[:,0] + l[0][3] * torch.tensor([[torch.cos(phi_M), torch.sin(phi_M), 0]]).to(device).repeat(B,1)
    e[:,4] = e[:,0] + l[0][4] * torch.tensor([[torch.cos(phi_R), torch.sin(phi_R), 0]]).to(device).repeat(B,1)
    e[:,5] = e[:,0] + l[0][5] * torch.tensor([[torch.cos(phi_L), torch.sin(phi_L), 0]]).to(device).repeat(B,1)

    # thumb
    e[:,1] = e[:,0] + l[0][1] * torch.cat([torch.cos(theta[:,0].unsqueeze(-1)) * torch.cos(pi/2 + phi[:,0].unsqueeze(-1)), \
        torch.cos(theta[:,0].unsqueeze(-1)) * torch.sin(pi / 2 + phi[:,0].unsqueeze(-1)), -torch.sin(theta[:,0].unsqueeze(-1))], dim=-1)
    e[:,6] = e[:,1] + l[1][6] * torch.cat([torch.cos(theta[:,0].unsqueeze(-1) + theta[:,1].unsqueeze(-1)) * torch.cos(pi/2 + phi[:,0].unsqueeze(-1)), \
        torch.cos(theta[:,0].unsqueeze(-1) + theta[:,1].unsqueeze(-1)) * torch.sin(pi/2 + phi[:,0].unsqueeze(-1)), -torch.sin(theta[:,0].unsqueeze(-1) + theta[:,1].unsqueeze(-1))], dim=-1)
    e[:,7] = e[:,6] + l[6][7] * torch.cat([torch.cos(theta[:,0].unsqueeze(-1) + theta[:,1].unsqueeze(-1) + theta[:,6].unsqueeze(-1)) * torch.cos(pi / 2 + phi[:,0:1]),\
        torch.cos(theta[:,0:1] + theta[:,1:2] + theta[:,6:7]) * torch.sin(pi / 2 + phi[:,0:1]), -torch.sin(theta[:,0:1] + theta[:,1:2] + theta[:,6:7])], dim=-1)
    
    # index finger
    e[:,8] = e[:,2] + l[2][8] * torch.cat([torch.cos(theta[:,2].unsqueeze(-1)) * torch.cos(phi_I + phi[:,2].unsqueeze(-1)), \
        torch.cos(theta[:,2].unsqueeze(-1)) * torch.sin(phi_I + phi[:,2].unsqueeze(-1)), -torch.sin(theta[:,2].unsqueeze(-1))], dim=-1)
    e[:,9] = e[:,8] + l[8][9] * torch.cat([torch.cos(theta[:,2].unsqueeze(-1) + theta[:,8].unsqueeze(-1)) * torch.cos(phi_I + phi[:,2].unsqueeze(-1)), \
        torch.cos(theta[:,2].unsqueeze(-1) + theta[:,8].unsqueeze(-1)) * torch.sin(phi_I + phi[:,2].unsqueeze(-1)), -torch.sin(theta[:,2].unsqueeze(-1) + theta[:,8].unsqueeze(-1))], dim=-1)
    e[:,10] = e[:,9] + l[9][10] * torch.cat([torch.cos(theta[:,2].unsqueeze(-1) + theta[:,8].unsqueeze(-1) + theta[:,9].unsqueeze(-1)) * torch.cos(phi_I + phi[:,2:3]),\
        torch.cos(theta[:,2:3] + theta[:,8:9] + theta[:,9:10]) * torch.sin(phi_I+ phi[:,2:3]), -torch.sin(theta[:,2:3] + theta[:,8:9] + theta[:,9:10])], dim=-1)

    # middle finger
    e[:,11] = e[:,3] + l[3][11] * torch.cat([torch.cos(theta[:,3].unsqueeze(-1)) * torch.cos(phi_M + phi[:,3].unsqueeze(-1)), \
        torch.cos(theta[:,3].unsqueeze(-1)) * torch.sin(phi_M + phi[:,3].unsqueeze(-1)), -torch.sin(theta[:,3].unsqueeze(-1))], dim=-1)
    e[:,12] = e[:,11] + l[11][12] * torch.cat([torch.cos(theta[:,3].unsqueeze(-1) + theta[:,11].unsqueeze(-1)) * torch.cos(phi_M + phi[:,3].unsqueeze(-1)), \
        torch.cos(theta[:,3].unsqueeze(-1) + theta[:,11].unsqueeze(-1)) * torch.sin(phi_M + phi[:,3].unsqueeze(-1)), -torch.sin(theta[:,3].unsqueeze(-1) + theta[:,11].unsqueeze(-1))], dim=-1)
    e[:,13] = e[:,12] + l[12][13] * torch.cat([torch.cos(theta[:,3].unsqueeze(-1) + theta[:,11].unsqueeze(-1) + theta[:,12].unsqueeze(-1)) * torch.cos(phi_M + phi[:,3:4]),\
        torch.cos(theta[:,3:4] + theta[:,11:12] + theta[:,12:13]) * torch.sin(phi_M + phi[:,3:4]), -torch.sin(theta[:,3:4] + theta[:,11:12] + theta[:,12:13])], dim=-1)
    
    # ring finger
    e[:,14] = e[:,4] + l[4][14] * torch.cat([torch.cos(theta[:,4].unsqueeze(-1)) * torch.cos(phi_R + phi[:,4].unsqueeze(-1)), \
        torch.cos(theta[:,4].unsqueeze(-1)) * torch.sin(phi_R + phi[:,4].unsqueeze(-1)), -torch.sin(theta[:,4].unsqueeze(-1))], dim=-1)
    e[:,15] = e[:,14] + l[14][15] * torch.cat([torch.cos(theta[:,4].unsqueeze(-1) + theta[:,14].unsqueeze(-1)) * torch.cos(phi_R + phi[:,4].unsqueeze(-1)), \
        torch.cos(theta[:,4].unsqueeze(-1) + theta[:,14].unsqueeze(-1)) * torch.sin(phi_R + phi[:,4].unsqueeze(-1)), -torch.sin(theta[:,4].unsqueeze(-1) + theta[:,14].unsqueeze(-1))], dim=-1)
    e[:,16] = e[:,15] + l[15][16] * torch.cat([torch.cos(theta[:,4].unsqueeze(-1) + theta[:,14].unsqueeze(-1) + theta[:,15].unsqueeze(-1)) * torch.cos(phi_R + phi[:,4:5]),\
        torch.cos(theta[:,4:5] + theta[:,14:15] + theta[:,15:16]) * torch.sin(phi_R + phi[:,4:5]), -torch.sin(theta[:,4:5] + theta[:,14:15] + theta[:,15:16])], dim=-1)
    
    # little finger
    e[:,17] = e[:,5] + l[5][17] * torch.cat([torch.cos(theta[:,5].unsqueeze(-1)) * torch.cos(phi_L + phi[:,5].unsqueeze(-1)), \
        torch.cos(theta[:,5].unsqueeze(-1)) * torch.sin(phi_L + phi[:,5].unsqueeze(-1)), -torch.sin(theta[:,5].unsqueeze(-1))], dim=-1)
    e[:,18] = e[:,17] + l[17][18] * torch.cat([torch.cos(theta[:,5].unsqueeze(-1) + theta[:,17].unsqueeze(-1)) * torch.cos(phi_L + phi[:,5].unsqueeze(-1)), \
        torch.cos(theta[:,5].unsqueeze(-1) + theta[:,17].unsqueeze(-1)) * torch.sin(phi_L + phi[:,5].unsqueeze(-1)), -torch.sin(theta[:,5].unsqueeze(-1) + theta[:,17].unsqueeze(-1))], dim=-1)
    e[:,19] = e[:,18] + l[18][19] * torch.cat([torch.cos(theta[:,5].unsqueeze(-1) + theta[:,17].unsqueeze(-1) + theta[:,18].unsqueeze(-1)) * torch.cos(phi_L + phi[:,5:6]),\
        torch.cos(theta[:,5:6] + theta[:,17:18] + theta[:,18:19]) * torch.sin(phi_L + phi[:,5:6]), -torch.sin(theta[:,5:6] + theta[:,17:18] + theta[:,18:19])], dim=-1)
    
    return e

if __name__ == '__main__':
    test_data = torch.rand(10,5,4)
    r = angle2cord(test_data)   
 
