import tensorflow as tf
import glob 
import os 

try:
    from torch.utils.tensorboard import SummaryWriter

    class PytorchLogger(object):
        def __init__(self, args):
            self.directory = os.path.join(args.log_dir, args.dataset, args.checkname)

            runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
            exist_run_ids = sorted([int(r.split('_')[-1]) for r in runs])
            run_id = exist_run_ids[-1] + 1 if runs else 0

            self.log_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            self.writer = SummaryWriter(self.log_dir)

        def add_images_with_bboxes(self, tag, images, bbox, step, labels=None, dataformats='CHW'):
            '''bbox:  N x 4 (xmin, ymin, xmax, ymax), absolute values'''
            self.writer.add_image_with_boxes(tag, images, bbox, step, dataformats=dataformats)
            self.writer.flush()
           
        def list_of_scalars_summary(self, tag_value_pairs, step):
            for tag, value in tag_value_pairs:
                self.writer.add_scalar(tag, value, step)
                self.writer.flush()

        def add_image(self, tag, image, step):
            self.writer.add_image(tag, image, step)
            self.writer.flush()

        def add_images(self, tag, images, step):
            self.writer.add_images(tag, images, step)
            self.writer.flush()

except:
    pass



try:
    tf_version = tf.__version__
except:
    tf_version = tf.VERSION


if int(tf_version[0]) < 2:
    class Logger(object):
        def __init__(self, log_dir):
            """Create a summary writer logging to log_dir."""
            self.writer = tf.summary.FileWriter(log_dir)

        def scalar_summary(self, tag, value, step):
            """Log a scalar variable."""
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            self.writer.add_summary(summary, step)

        def list_of_scalars_summary(self, tag_value_pairs, step):
            """Log scalar variables."""
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value) for tag, value in tag_value_pairs])
            self.writer.add_summary(summary, step)

        def image_summary(self, tag, value, step):
            summary = tf.Summary.image(tag, value, step=step)
            self.writer.add_summary(summary, step)

else:
    class Logger(object):
        def __init__(self, log_dir):
            """Create a summary writer logging to log_dir."""
            self.writer = tf.summary.create_file_writer(log_dir)

        def scalar_summary(self, tag, value, step):
            """Log a scalar variable."""
            with self.writer.as_default():
                tf.summary.scalar(tag, value, step=step)
                self.writer.flush()
            # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            # self.writer.add_summary(summary, step)

        def list_of_scalars_summary(self, tag_value_pairs, step):
            """Log scalar variables."""
            with self.writer.as_default():
                for tag, value in tag_value_pairs:
                    tf.summary.scalar(tag, value, step=step)
                self.writer.flush()
            # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value) for tag, value in tag_value_pairs])
            # self.writer.add_summary(summary, step)
        
        def image_summary(self, tag, value, step):
            with self.writer.as_default():
                tf.summary.image(tag, value, step=step)
                self.writer.flush()
    
        def bbox_summary(self, tag, images, boxes,  colors):
            with self.writer.as_default():
                tf.summary.image(tag, value, step=step)
                self.writer.flush()

