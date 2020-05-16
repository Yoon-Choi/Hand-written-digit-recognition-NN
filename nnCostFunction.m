function [J, grad] = nnCostFunction(nn_params, ...
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
X = [ones(m,1) X]; % bias �߰� X = 5000 x 401 

activate2 = zeros(size(Theta1,1),m); % size(Theta1,1=��) = 25   X 5000 
activate3 = zeros(size(Theta2,1),m); %size(Theta2,1 = ��) = 10 X 5000

act2 = zeros(size(Theta1,1),1); % 25 X 1 �̺κ� �� �ϴ°ɱ� ? 
act3 = zeros(size(Theta2,1),1); % 10 X 1 �̺κ� �� �ϴ°ɱ� ? 
%%
% �̺κ� �� activate�� act�� a�� ���� �����ϴ°ɱ� ? 
activate2 = sigmoid (Theta1 * X'); %  activate2 = 25 X 5000 25 X 401  * 401 x 5000 
activate3 = sigmoid (Theta2 * [ones(1,m) ; activate2]); % activate3 =  10 X 5000 
% [ones(1,m) = 1�� 5000��  ; activate2] =�� ���ٿ� bias �߰� = 26 X 5000 �� �� 
%Theta2 = 10 X 26  * 26 X 5000  = activate3  = 10 X 5000 
output = 1:num_labels; % �̺κ� ����ΰ� ? �Ƹ��� �´µ� ?=== 1~ 10  10X1����� �ƴϰ� 1X 10 ����̴�.                     
% �̺κ� recode �� ���� ���̴µ� ����ִ°ɱ� ? "equality ������"�� ���� �ִ°ǰ� ? 


for i = 1:m % 5000
    Y = y(i) == output; % recoded  Y = 1x 10
        % ==�����ڴ� array�� �����Ѵ�. 
        % y(i) == output �� �ᵵ �ڿ������� recode �ȴ�. 
        % y(i)�� 10���� �Ѱ� �̰�  output�� 1x10 ����̴�. ����� 1x 10��ķ� ���� �� �Ͽ�
        %���� �κ���  logical array �� �����Ѵ�. ���� �κ��� 1�� ������ 0����. 

    J = J - Y * log (activate3(:, i)); %  J = 1 X 1
            % Y = 1x 10     activate3(:,i) =  10 X 1

    
    J = J - (1 - Y) * log (1 - activate3(:, i));
    % - 1/m �� - ��ȣ�� " ������ ��ȣ ��"�� ����־ -�� �� ��. 
end % for���� ���� �� �� ���� J= cost ����  1���� ���� �� ������ �ȴ�.               
J = J / m;  % �̺κ� �� �̷��� ������ ? === ���� J�� 1/m�� �Ѱ�. -��ȣ�� for�� �ȿ� ���� �ڵ带 ®��. 
reg =  lambda / 2 / m ...
    * (sum(sum(Theta1(:, 2:end) .* Theta1(:, 2:end))) ...
    + sum(sum(Theta2(:, 2:end) .* Theta2(:, 2:end))));

J = J + reg;     
%%
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
    a_1 = X(i,:)'; % a_1 = 401 X 1   % bias �߰��� X = 5000 x 401 
    z_2 = Theta1 * a_1; %z_2 = 25 X 1
    %a_2 = sigmoid(z_2);
    a_2 = [1 ; sigmoid(z_2)];
    z_3 = Theta2 * a_2;
    a_3 = sigmoid(z_3);
        
    d_3 = a_3 - (y(i) == output)';
    d_2 = (Theta2' * d_3) .* sigmoidGradient([1; z_2]);
    %d_2 = d_2(2:end);
    Delta_2 = Delta_2 + d_3 * a_2';
    Delta_1 = Delta_1 + d_2(2:end) * a_1';
end

Theta2_grad = Delta_2 ./ m;
Theta1_grad = Delta_1 ./ m;
%% regularization
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