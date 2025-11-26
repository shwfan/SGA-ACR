import framework
import tasks
from config.default import register_args

checkpoint_path = None

def main():
    helper = framework.helpers.TrainingHelper(register_args=register_args)
    task = tasks.RLTaskCRF(helper, checkpoint_path=checkpoint_path)
    task.train()
    helper.finish()

if __name__ == '__main__':
    main()

