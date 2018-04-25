import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.*;
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
        Integer N = Integer.parseInt(args[0]);
        String input = args[1];
        String output = args[2]+"/tmp0";
        Integer i = 0;

        Configuration conf = getConf();
        conf.set("N", N.toString());
        conf.set("alpha", "0.15");
        conf.set("Iter", "0");
        Job job = GetJobConf(conf, input, output);
        if (!job.waitForCompletion(true))
            return 1;

        conf.set("Iter", "1");
        while (job.getCounters().findCounter("COMMON_COUNTERS", "LowChanges").getValue() != N ) {
            System.out.println("Iter: "+i+", LowChanges/N: "+
                    job.getCounters().findCounter("COMMON_COUNTERS", "LowChanges").getValue()+
                    "/"+N);

            job.getCounters().findCounter("COMMON_COUNTERS", "LowChanges").setValue(0);
            i++;
            input = output+"/part*";
            output = args[2]+"/tmp"+i;
            job = GetJobConf(getConf(), input, output);
            if (!job.waitForCompletion(true))
                return 1;
        }

        return 0;
    }

    private Job GetJobConf(Configuration conf, String input, String out_dir) throws IOException {
        Job job = Job.getInstance(conf);
        job.setJarByClass(PageRank.class);
        job.setJobName(PageRank.class.getCanonicalName());

        FileInputFormat.addInputPath(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path(out_dir));

        job.setMapperClass(PageRank.PRMapper.class);
        job.setReducerClass(PageRank.PRReducer.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);

        return job;
    }

    public static class PRMapper extends Mapper<LongWritable, Text, IntWritable, Text>
    {
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
//            System.out.println(value.toString());
            String[] split = value.toString().split("\t");
//            System.out.println(split.length);
            IntWritable node_id;
            Double rank;
            String[] node_list;
            Integer N = Integer.parseInt(context.getConfiguration().get("N"));

            node_id = new IntWritable(Integer.parseInt(split[0]));

            if (context.getConfiguration().get("Iter").equals("0")) {
                rank = 1.0 / N;
                if (split.length == 1) {
                    node_list = new String[0];
                    context.write(node_id, new Text("S"));
                } else {
                    node_list = split[1].split(" ");
                    context.write(node_id, new Text("S"+split[1]));
                }
            } else {
                rank = Double.parseDouble(split[1]);
                if (split.length == 2) {
                    node_list = new String[0];
                    context.write(node_id, new Text("S"));
                } else {
                    node_list = split[2].split(" ");
                    context.write(node_id, new Text("S"+split[2]));
                }
            }

            context.write(node_id, new Text("O"+rank));

            if (node_list.length == 0){
                for (Integer i=0; i < N; i++){
                    if (i != node_id.get()){
                        Double tmp = rank/(N-1);
                        context.write(new IntWritable(i), new Text(tmp.toString()));
                    }
                }
            } else {
                for (String node : node_list){
                    Double tmp = rank/node_list.length;
                    context.write(new IntWritable(Integer.parseInt(node)),
                                  new Text(tmp.toString()));
                }
            }


        }
    }

    public static class PRReducer extends Reducer<IntWritable, Text, IntWritable, Text> {
        @Override
        protected void reduce(IntWritable node_id, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            Text graph = new Text();
            Double old_rank = 0.0;
            Double rank = 0.0;
            Double alpha = Double.parseDouble(context.getConfiguration().get("alpha"));
            Integer N = Integer.parseInt(context.getConfiguration().get("N"));
            for (Text val: values) {
                if (val.charAt(0) == 'S'){
                    graph = new Text(val.toString().substring(1));
                } else if (val.charAt(0) == 'O'){
                    old_rank = Double.parseDouble(val.toString().substring(1));
                } else {
                    rank += Double.parseDouble(val.toString());
                }
            }

            rank = alpha/N + (1.0-alpha)*rank;

            if (Math.abs(old_rank - rank) < 0.0001)
                context.getCounter("COMMON_COUNTERS", "LowChanges").increment(1);

            context.write(node_id, new Text(rank + "\t" + graph.toString()));
        }
    }
}
