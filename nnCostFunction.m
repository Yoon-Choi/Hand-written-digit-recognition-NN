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
%%%%%궁금한 사항을모아둠. "이부분" 으로 검색해서 찾아서 정리한다. 
% X = 5000 X 400
% Theta1 = 25 X 401
% Theta2 = 10 X 26
% activate1 = 25 X 5000
% activate2 = 10 X 5000
%%
X = [ones(m,1) X]; % bias 추가 X = 5000 x 401 

activate2 = zeros(size(Theta1,1),m); % size(Theta1,1=행) = 25   X 5000 
activate3 = zeros(size(Theta2,1),m); %size(Theta2,1 = 행) = 10 X 5000

act2 = zeros(size(Theta1,1),1); % 25 X 1 이부분 왜 하는걸까 ? 
act3 = zeros(size(Theta2,1),1); % 10 X 1 이부분 왜 하는걸까 ? 
%%
% 이부분 왜 activate와 act와 a를 따로 지정하는걸까 ? 
activate2 = sigmoid (Theta1 * X'); %  activate2 = 25 X 5000 25 X 401  * 401 x 5000 
activate3 = sigmoid (Theta2 * [ones(1,m) ; activate2]); % activate3 =  10 X 5000 
% [ones(1,m) = 1행 5000열  ; activate2] =맨 윗줄에 bias 추가 = 26 X 5000 이 됨 
%Theta2 = 10 X 26  * 26 X 5000  = activate3  = 10 X 5000 
output = 1:num_labels; % 이부분 행렬인가 ? 아마도 맞는듯 ?=== 1~ 10  10X1행렬이 아니고 1X 10 행렬이다.                     
% 이부분 recode 가 없어 보이는데 어디있는걸까 ? "equality 연산자"와 관련 있는건가 ? 


for i = 1:m % 5000
    Y = y(i) == output; % recoded  Y = 1x 10
        % ==연산자는 array를 리턴한다. 
        % y(i) == output 만 써도 자연스럽게 recode 된다. 
        % y(i)는 10진수 한개 이고  output은 1x10 행렬이다. 결과는 1x 10행렬로 서로 비교 하여
        %같은 부분을  logical array 로 리턴한다. 참인 부분을 1로 거짓을 0으로. 

    J = J - Y * log (activate3(:, i)); %  J = 1 X 1
            % Y = 1x 10     activate3(:,i) =  10 X 1

    
    J = J - (1 - Y) * log (1 - activate3(:, i));
    % - 1/m 의 - 부호를 " 수식의 괄호 안"에 집어넣어서 -가 된 것. 
end % for문이 끝나 고 난 뒤의 J= cost 값은  1개의 숫자 로 나오게 된다.               
J = J / m;  % 이부분 왜 이렇게 했을까 ? === 최종 J에 1/m을 한것. -부호는 for문 안에 들어가게 코드를 짰다. 
reg =  lambda / 2 / m ...
    * (sum(sum(Theta1(:, 2:end) .* Theta1(:, 2:end))) ...
    + sum(sum(Theta2(:, 2:end) .* Theta2(:, 2:end))));

J = J + reg;     
%%
Delta_2 = zeros(size(Theta2)); % Delta_2 =  10 X 26
Delta_1 = zeros(size(Theta1)); % Delta_1 = 25 X 401
a_1 = zeros(size(Theta1, 2), 1); % a_1 = 401 X 1
% 이부분 저 밑의 a_1 이랑 다른걸까 ?

z_2 = zeros(size(Theta1, 1), 1); % Theta1의 행 z_2 = 25 X 1

%Theta2 = 10 X 26 
a_2 = zeros(size(Theta2, 2), 1); % Theta2의 렬 a_2 = 26 X 1
z_3 = zeros(size(Theta2, 1), 1); % Theta2의 행 z_3 = 10 X 1
a_3 = zeros(size(Theta2, 1), 1); % Theta2의 행 a_3 = 10 X 1
d_3 = zeros(size(Theta2,1), 1); % Theta2의 행 d_3 = 10 X 1
d_2 = zeros(size(Theta2,2),1); % Theta2의 렬 d_2 = 26 X 1
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
    a_1 = X(i,:)'; % a_1 = 401 X 1   % bias 추가된 X = 5000 x 401 
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
  % 염두 해 두어야 할 것 = ""i, j의 역할이 고정되어있지 않다는것. ""
  %Theta1_grad = 25 X 401 
for i = 1:size(Theta1_grad,1)% 행 25 
    for j = 2:size(Theta1_grad,2)% 렬 401 
           Theta1_grad(i,j) = Theta1_grad(i,j) + Theta1(i,j) * lambda / m;
    end
end

  %Theta2_grad = 10 X 26 
for i = 1:size(Theta2_grad,1) % 행 10 
    for j = 2:size(Theta2_grad,2)% 렬 26 
        Theta2_grad(i,j) = Theta2_grad(i,j) + Theta2(i,j) * lambda / m;
    end
end
%%
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
%grad = partial derivative of J = backpropagation = grad 

end
%============================================================