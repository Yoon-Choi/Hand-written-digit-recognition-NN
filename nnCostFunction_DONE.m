function [J, grad] = nnCostFunction_DONE(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1)); % Theta1_grad = 25 X 401
Theta2_grad = zeros(size(Theta2)); %Theta2_grad = 10 X 26 

%%
%%%%%�ñ��� ��������Ƶ�. "�̺κ�" ���� �˻��ؼ� ã�Ƽ� �����Ѵ�. 
% X = 5000 X 400
% Theta1 = 25 X 401
% Theta2 = 10 X 26
% activate1 = 25 X 5000
% activate2 = 10 X 5000
%%

activate0 = ones(m,1); % bias
X = [activate0, X]; % bias �߰� X = 5000 x 401 
%activate1 = X; % input layer 
activate2 = zeros(size(Theta1,1),m); % size(Theta1,1=��) = 25   X 5000 
activate3 = zeros(size(Theta2,1),m); %size(Theta2,1 = ��) = 10 X 5000

act2 = zeros(size(Theta1,1),1); % 25 X 1 �̺κ� �� �ϴ°ɱ� ? 
act3 = zeros(size(Theta2,1),1); % 10 X 1 �̺κ� �� �ϴ°ɱ� ? 
%%
% �̺κ� �� activate�� act�� a�� ���� �����ϴ°ɱ� ? 
activate2 = sigmoid (Theta1 * X'); %  activate2 = 25 X 5000  ----- 25 X 401  * 401 x 5000 
activate3 = sigmoid (Theta2 * [ones(1,m) ; activate2]); % activate3 =  10 X 5000 
% [ones(1,m) = 1�� 5000��  ; activate2] =�� ���ٿ� bias �߰� = 26 X 5000 �� �� 
%Theta2 = 10 X 26  * 26 X 5000  = activate3  = 10 X 5000 
k = num_labels;
K = 1: k; % 1 X 10  % 10���� output             

temp = 0;
for i = 1: m % 5000
    Y = y(i) == K; % recoded  Y = 1x 10
        % ==�����ڴ� array�� �����Ѵ�. 
        % y(i) == output �� �ᵵ �ڿ������� recode �ȴ�. 
        % y(i)�� 10���� �Ѱ� �̰�  output�� 1x10 ����̴�. ����� 1x 10��ķ� ���� �� �Ͽ�
        %���� �κ���  logical array �� �����Ѵ�. ���� �κ��� 1�� ������ 0����. 
    h = activate3(:, i);   %hypothesis  10 X 1
    temp = temp + (Y * log(h) + (1 - Y) * log (1 - h));    
end % for���� ���� �� �� ���� temp = cost ����  1���� ���� �� ������ �ȴ�. 
J = (-1/ m).* temp;  
reg =  lambda / (2 * m) ...
    * (sum(sum(Theta1(:, 2:end) .* Theta1(:, 2:end))) ...
    + sum(sum(Theta2(:, 2:end) .* Theta2(:, 2:end))));
J = J + reg;     
%% ���� ��ĭ ���� 
Delta_2 = zeros(size(Theta2)); % Delta_2 =  10 X 26
Delta_1 = zeros(size(Theta1)); % Delta_1 = 25 X 401


a_1 = zeros(size(Theta1, 2), 1); % a_1 = 401 X 1
% �̺κ� �� ���� a_1 �̶� �ٸ��ɱ� ?

z_2 = zeros(size(Theta1, 1), 1); % Theta1�� �� z_2 = 25 X 1

%Theta2 = 10 X 26 
a_2 = zeros(size(Theta2, 2), 1); % Theta2�� �� a_2 = 26 X 1
z_3 = zeros(size(Theta2, 1), 1); % Theta2�� �� z_3 = 10 X 1

a_3 = zeros(size(Theta2, 1), 1); % Theta2�� �� a_3 = 10 X 1

d_3 = zeros(size(Theta2,1), 1); % Theta2�� �� d_3 = 10 X 1
d_2 = zeros(size(Theta2,2),1); % Theta2�� �� d_2 = 26 X 1
% d_1 = ����. input�̶� 
%% 
% a_1 = 401 X 1
% z_2 = 25 X 1
% a_2 = 26 X 1
% z_3 = 10 X 1
% a_3 = 10 X 1
% d_3 = 10 X 1
% d_2 = 26 X 1
% d_2 = 25 X 1
% Delta_2 = 10 X 26
% Delta_1 = 25 X 401

%%
for i = 1:m  % m = 5000 
    %a_1 = X(i,:)';
    
    %lecture9 �����̵� 6������ �κ� ����
    a_1 = X(i,:)'; % a_1 = 401 X 1   % bias �߰��� X = 5000 x 401 
    z_2 = Theta1 * a_1; %z_2 = 25 X 1 
    a_2 = [1 ; sigmoid(z_2)]; % bias�߰�  %a_2 = sigmoid(z_2);
    z_3 = Theta2 * a_2;
    a_3 = sigmoid(z_3); % hypothesis 
    
    %lecture9 �����̵� 7������ �κ� ����     
    d_3 = a_3 - Y'; % Y' = 10X1    a_3 = 10X1
    d_2 = (Theta2' * d_3) .*...
        sigmoidGradient([1; z_2]); % bias�߰� 
    
    %lecture9 �����̵� 8������ �κ� ����. �� �������� ��ü �帧�� ���� �κ��� ��������� Delta�κ� ������ �����ϸ�
    % �����̵��� Delta ���ĺο��� l+1 = 3(��ü���̾�L)  �� �ǵ��� Delta �� �����Ѵ�. 
    Delta_2 = Delta_2 + d_3 * a_2'; % Delta_������ ���Ŀ��� �ﰢ���κ� 
    Delta_1 = Delta_1 + d_2(2:end) * a_1'; % bias ����
end

%% ���Ŀ��� D�κп� �ش��ϴ� �� ��  
 %%%% �ؿ� �� j = 0 �κ� ���� 
% Theta2_grad = ���Ŀ��� D�κп� �ش�
Theta2_grad = Delta_2 ./ m;  % regularization term�� ���� ���� Ȯ���� �� �ִ�. 
Theta1_grad = Delta_1 ./ m;

 %%%%���� �� j != 0 �κ� ���� 
% regularization term���� ������ �κ� 
% ���� �� �ξ�� �� �� = ""i, j�� ������ �����Ǿ����� �ʴٴ°�. ""
%Theta1_grad = 25 X 401 
for i = 1:size(Theta1_grad,1)% �� 25 
    for j = 2:size(Theta1_grad,2)% �� 401 
           Theta1_grad(i,j) = Theta1_grad(i,j) + Theta1(i,j) * lambda / m;
    end
end

  %Theta2_grad = 10 X 26 
for i = 1:size(Theta2_grad,1) % �� 10 
    for j = 2:size(Theta2_grad,2)% �� 26 
        Theta2_grad(i,j) = Theta2_grad(i,j) + Theta2(i,j) * lambda / m;
    end
end
%%
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
%grad = partial derivative of J = backpropagation = grad 

end
%============================================================