# RL-for-All

기존의 Sun Kim 교수님이 [모두를 위한 RL강좌](https://youtube.com/playlist?list=PLlMkM4tgfjnKsCWav-Z2F-MMFRx-2gMGG)에서 소개하신 코드가 텐서플로우 1.x 버전이기에 현재 그대로 따라하기에 무리가 있습니다. 그래서 텐서플로우 2.x 버전에서 권장하는 케라스 모델을 이용해 관련 코드들을 새롭게 바꾸었습니다. 텐서플로우를 사용하는 Lab 6부터 코드를 수정하였습니다. 버전 변경 후 코드 구조가 바뀌어 느려지는 현상을 수정하기 위해 DQN 모델은 학습 부분을 전체적으로 손 봐서 최대한 실행시간을 줄였습니다.

## 수정 사항

### 1. 텐서플로우 버전 변경

`tf.placeholder`를 사용하지 않고 `Sequential()`을 사용하였습니다. 케라스 모델을 사용하였으므로 나머지 함수들도 맞게 바꾸었습니다.

### 2. 최적화

Q-Network 예제처럼 DQN 예제를 무작정 텐서플로우로 바꾸면 학습 시간이 너무 길어졌습니다. 이 문제는 텐서플로우 버전을 올리면서 작동하는 방법이 바뀌었기 때문에 생긴 문제였습니다. 

```
def replay_train(self, network):       
    x_stack = np.empty(0).reshape(0, network.input_size)
    y_stack = np.empty(0).reshape(0, network.output_size)
    for _ in range(50):
        batch = random.sample(self.buffer, 10)
        for state, action, reward, next_state, done in batch:
            q = network.predict(state)
            if done:
                q[0, action] = reward
            else:
                q[0, action] = reward + self.dis * np.max(network.predict(next_state))
            y_stack = np.vstack([y_stack, q])
            x_stack = np.vstack([x_stack, state])
    return network.fit(x_stack, y_stack)
```

위처럼 Q-Network때 사용한 방식으로 코드를 짤 경우 실행 시간은 `replay_train` 함수가 한 번 호출될 때마다 최소 약 45초가 걸립니다. 대부분은 `network.predict`를 두 번이나 쓰면서 생긴 문제입니다. 여기서 Q값을 받아오는 `q = network.predict(state)`는 `main` 메소드에서도 한번 호출합니다. 이 Q값은 `replay_train` 함수를 호출하기 전에는 바뀌지 않습니다. 그래서 메모리에 넣을 때 `q`도 같이 넣어 `predict` 함수를 호출하는 횟수를 한 번 줄여 실행 시간을 평균 30초로 만들었습니다.
