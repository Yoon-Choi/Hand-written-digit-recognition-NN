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
%%%%%궁금한 사항을모아둠. "이부분" 으로 검색해서 찾아서 정리한다. 
% X = 5000 X 400
% Theta1 = 25 X 401
% Theta2 = 10 X 26
% activate1 = 25 X 5000
% activate2 = 10 X 5000
%%

activate0 = ones(m,1); % bias
X = [activate0, X]; % bias 추가 X = 5000 x 401 
%activate1 = X; % input layer 
activate2 = zeros(size(Theta1,1),m); % size(Theta1,1=행) = 25   X 5000 
activate3 = zeros(size(Theta2,1),m); %size(Theta2,1 = 행) = 10 X 5000

act2 = zeros(size(Theta1,1),1); % 25 X 1 이부분 왜 하는걸까 ? 
act3 = zeros(size(Theta2,1),1); % 10 X 1 이부분 왜 하는걸까 ? 
%%
% 이부분 왜 activate와 act와 a를 따로 지정하는걸까 ? 
activate2 = sigmoid (Theta1 * X'); %  activate2 = 25 X 5000  ----- 25 X 401  * 401 x 5000 
activate3 = sigmoid (Theta2 * [ones(1,m) ; activate2]); % activate3 =  10 X 5000 
% [ones(1,m) = 1행 5000열  ; activate2] =맨 윗줄에 bias 추가 = 26 X 5000 이 됨 
%Theta2 = 10 X 26  * 26 X 5000  = activate3  = 10 X 5000 
k = num_labels;
K = 1: k; % 1 X 10  % 10진수 output             

temp = 0;
for i = 1: m % 5000
    Y = y(i) == K; % recoded  Y = 1x 10
        % ==연산자는 array를 리턴한다. 
        % y(i) == output 만 써도 자연스럽게 recode 된다. 
        % y(i)는 10진수 한개 이고  output은 1x10 행렬이다. 결과는 1x 10행렬로 서로 비교 하여
        %같은 부분을  logical array 로 리턴한다. 참인 부분을 1로 거짓을 0으로. 
    h = activate3(:, i);   %hypothesis  10 X 1
    temp = temp + (Y * log(h) + (1 - Y) * log (1 - h));    
end % for문이 끝나 고 난 뒤의 temp = cost 값은  1개의 숫자 로 나오게 된다. 
J = (-1/ m).* temp;  
reg =  lambda / (2 * m) ...
    * (sum(sum(Theta1(:, 2:end) .* Theta1(:, 2:end))) ...
    + sum(sum(Theta2(:, 2:end) .* Theta2(:, 2:end))));
J = J + reg;     
%% 변수 빈칸 세팅 
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
% d_1 = 없음. input이라 
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
    
    %lecture9 슬라이드 6페이지 부분 구현
    a_1 = X(i,:)'; % a_1 = 401 X 1   % bias 추가된 X = 5000 x 401 
    z_2 = Theta1 * a_1; %z_2 = 25 X 1 
    a_2 = [1 ; sigmoid(z_2)]; % bias추가  %a_2 = sigmoid(z_2);
    z_3 = Theta2 * a_2;
    a_3 = sigmoid(z_3); % hypothesis 
    
    %lecture9 슬라이드 7페이지 부분 구현     
    d_3 = a_3 - Y'; % Y' = 10X1    a_3 = 10X1
    d_2 = (Theta2' * d_3) .*...
        sigmoidGradient([1; z_2]); % bias추가 
    
    %lecture9 슬라이드 8페이지 부분 구현. 이 페이지는 전체 흐름에 대한 부분을 적어뒀지만 Delta부분 구현만 생각하면
    % 슬라이드의 Delta 수식부에서 l+1 = 3(전체레이어L)  이 되도록 Delta 를 지정한다. 
    Delta_2 = Delta_2 + d_3 * a_2'; % Delta_변수는 수식에서 삼각형부분 
    Delta_1 = Delta_1 + d_2(2:end) * a_1'; % bias 제외
end

%% 수식에서 D부분에 해당하는 두 줄  
 %%%% 밑에 줄 j = 0 부분 수식 
% Theta2_grad = 수식에서 D부분에 해당
Theta2_grad = Delta_2 ./ m;  % regularization term이 없는 것을 확인할 수 있다. 
Theta1_grad = Delta_1 ./ m;

 %%%%위에 줄 j != 0 부분 수식 
% regularization term까지 구현된 부분 
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