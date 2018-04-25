import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class PageRank extends Configured implements Tool {

    static public void main(String[] args) throws Exception {
        int ret = ToolRunner.run(new PageRank(), args);
        System.exit(ret);
    }

    @Override
    public int run(String[] args) throws Exception {
        String input = args[0];
        String output = args[1]+"/tmp0";
        Integer i = 0;
        Integer N = 0;

        Path pt = new Path(args[0]);
        FileSystem fs = FileSystem.get(getConf());
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fs.open(pt)));
        String line;
        while ((line = bufferedReader.readLine()) != null) {
            N++;
        }

//        Job job = GetJobConf(getConf(), args[0], args[1], args[2]);
//        if (job.waitForCompletion(true)) {
//
//        }
//        return job.waitForCompletion(true) ? 0 : 1;
        System.out.println("N: " + N);
        return 0;
    }

    private Job GetJobConf(Configuration conf, String index_file_name, String input, String out_dir) throws IOException {
        conf.set("index_file", index_file_name);

        Job job = Job.getInstance(conf);
        job.setJarByClass(PageRank.class);
        job.setJobName(PageRank.class.getCanonicalName());

        FileInputFormat.addInputPath(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path(out_dir));

        job.setMapperClass(PageRank.PRMapper.class);
        job.setNumReduceTasks(0);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(LongArrayWritable.class);

        return job;
    }

    public static class PRMapper extends Mapper<LongWritable, Text, LongWritable, LongArrayWritable>
    {
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            context.getCounter("COMMON_COUNTERS", "MalformedUrls").setValue(0);
        }
    }
}
