function[features]=calculatefeatures(comp, ele_frac)
% *************************************************************************
%  Matlab Code to Compute 
% the average radius and its deviation, 
% the average melt temperature and its deviation, the average mixing enthalpy and its
%  deviation, and the ideal entropy,the average electronal negativity and
%  its deviation, the average VEC and its deviation, the radio of
%  Scorr/Sid, the bulk modulus and its deviation.
% *************************************************************************
% data1_ld = xlsread('FeatureExaction_properties.xlsx','C2:H84');       % read the database
% data2_ld = xlsread('FeatureExaction_mixingenthalpy.xlsx','C3:CG85');  % read the mixing enthalpy in the database
data1_ld = readmatrix('FeatureExaction_properties.xlsx', 'Range', 'C2:H84');
data2_ld = readmatrix('FeatureExaction_mixingenthalpy.xlsx', 'Range', 'C3:CG85');

radius = data1_ld(:,1);  % the atomic size
tm = data1_ld(:,2);      % the melting point
elec_nega = data1_ld(:,3); % the Pauling electronic negativity
vec = data1_ld(:,4);       % VEC
bulk = data1_ld(:,6);      % bulk modulus
enthalpy = data2_ld;   % the mixing enthalpy
ntotal = length(comp); % the number of elements
ele_size=zeros(ntotal,1);
ele_temp=zeros(ntotal,1);
ele_elec_nega=zeros(ntotal,1);
VEC=zeros(ntotal,1);
for i=1:ntotal
    ele_size(i)=radius(comp(i));
    ele_temp(i)=tm(comp(i));
    ele_elec_nega(i)=elec_nega(comp(i));
    VEC(i)=vec(comp(i));
end  
% *************************************************************************
%   Calculate the fractions of each element 
% *************************************************************************
sum=0;
for i=1:ntotal
    sum=sum+ele_frac(i);
end
ele_frac=ele_frac/sum; 
% *************************************************************************
%   Calculate the mean redius and delta  
% *************************************************************************
r_mean=0;                              
for  i=1:ntotal
    r_mean=r_mean+ele_size(i)*ele_frac(i);      %the average atomic radius
end
delta=0;
for  i=1:ntotal
    delta=delta+ele_frac(i)*(1-ele_size(i)/r_mean)^2;
end
delta=sqrt(delta);  % the delta

% *************************************************************************
%   Calculate the average melting temperature and its deviation 
% *************************************************************************
TM=0;
for i=1:ntotal
    TM=TM+ele_frac(i)*ele_temp(i); % the average melting temperature
end
DTM=0;
for i=1:ntotal
    DTM=DTM+ele_frac(i)*(ele_temp(i)-TM)^2;
end
DTM=sqrt(DTM); % the deviation of melting temperature


% *************************************************************************
%   Calculate the average electronic negativity and its deviation 
% *************************************************************************
Mean_elecnega=0;
for i=1:ntotal
    Mean_elecnega=Mean_elecnega+ele_elec_nega(i)*ele_frac(i);
end
D_elecnega=0;
for i=1:ntotal
    D_elecnega=D_elecnega+ele_frac(i)*(ele_elec_nega(i)-Mean_elecnega)^2;
end
D_elecnega=sqrt(D_elecnega);


% *************************************************************************
%   Calculate the average electronic negativity and its deviation 
% *************************************************************************
MVEC=0;
for i = 1:ntotal
    MVEC=MVEC+ele_frac(i)*VEC(i);
end
D_VEC=0;
for i = 1:ntotal
    D_VEC=D_VEC+ele_frac(i)*(VEC(i)-MVEC)^2;
end
D_VEC=sqrt(D_VEC);


% *************************************************************************
%   Calculate the total mixing enthalpy 
% *************************************************************************
ME=0;
for i=1:ntotal-1
    for j=i+1:ntotal
        ME=ME+4*ele_frac(i)*ele_frac(j)*enthalpy(comp(i),comp(j));
    end
end   %the mixing enthalpy


% *************************************************************************
%   Calculate the deviation of mixing enthaply 
% *************************************************************************
DME=0;
for i=1:ntotal-1
    for j=i+1:ntotal
        DME=DME+ele_frac(i)*ele_frac(j)*(enthalpy(comp(i),comp(j))-ME)^2;
    end
end
DME=sqrt(DME); %the deviation of mixing enthaply


% *************************************************************************
%   Calculate the ideal mixing entropy
% *************************************************************************
Sid=0;
for i=1:ntotal
    Sid=Sid-ele_frac(i)*log(ele_frac(i));
end


% *************************************************************************
%   Calculate average bulk modulus B and its standard deviation
% *************************************************************************
B_ave=0;
B=zeros(ntotal,1);
for i=1:ntotal
    B(i)=data1_ld(comp(i),6);
end
for i=1:ntotal
    B_ave=B_ave+ele_frac(i)*B(i);
end
D_Bulk=0;
for i = 1:ntotal
    D_Bulk=D_Bulk+ele_frac(i)*(B(i)-B_ave)^2;
end
D_Bulk=sqrt(D_Bulk);
B_ave=B_ave*1e9;


% *************************************************************************
%   Collecting all the features for machine learning
% *************************************************************************

features=zeros(13,1);
features(1)=r_mean; 
features(2)=delta;
features(3)=TM;
features(4)=DTM;
features(5)=ME;
features(6)=DME;
features(7)=Sid;
features(8)=Mean_elecnega;
features(9)=D_elecnega;
features(10)=MVEC;
features(11)=D_VEC;
features(12)=B_ave;
features(13)=D_Bulk;

features=features';

