import featurizer
import model
import obs_store

class QueryHandler():

    def __init__(self):
        self.model = model.Model('500_0.0001_32_lstm2x.pt')
        self.obs_store = obs_store.ObservationStore()

    def select_plan(self, messages):
        *plans, buffers = messages
        f_plan = featurizer.featurize(plans, buffers)
        i_plan = self.model.select_plan(f_plan)
        self.obs_store.stage(f_plan, i_plan)
        #self.obs_store.stage(f_plan, i_plan, self.gen_buffer_key(buffers))
        print("Selected=%d" % (i_plan))
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
        #self.obs_store.record(reward, self.gen_buffer_key(buffers))
        print("Reward=%f" % (reward))
        return

    def gen_buffer_key(self, buffers):
        print(buffers)
        return str(buffers['question_pkey'])+'.'+str(buffers['question'])