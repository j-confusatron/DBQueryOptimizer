import featurizer
import model
import obs_store
import json

class QueryHandler():

    def __init__(self):
        self.model = model.Model('model.pt')
        self.obs_store = obs_store.ObservationStore()

    def select_plan(self, messages, debug=False):
        *plans, buffers = messages
        if debug:
            plans = plans[0]
        f_plan = featurizer.featurize(plans, buffers)
        i_plan, method = self.model.select_plan(f_plan)
        self.obs_store.stage(f_plan, i_plan)
        print("Selected=%d (%s)" % (i_plan, method))
        return i_plan

    def predict(self, messages):
        # Might come back and properly implement this later.
        # It is not mission critical.
        #*plan, buffers = messages
        #f_plan = featurizer.featurize(plan)pytho
        #self.model.predict(f_plan)
        return 0

    def load_model(self, path):
        # Unneeded
        return

    def store(self, plan, buffers, obs_reward):
        reward = obs_reward['reward']
        self.obs_store.record(reward)
        print("Reward=%f" % (reward))
        return

    def gen_buffer_key(self, buffers):
        print(buffers)
        return str(buffers['question_pkey'])+'.'+str(buffers['question'])


# Debug
if __name__ == '__main__':
    with open('misc/plan.json', 'r') as fp:
        plans = json.load(fp)
    with open('misc/buffer.json', 'r') as fp:
        buffers = json.load(fp)
    qh = QueryHandler()
    i_plan = qh.select_plan((plans, buffers), debug=True)