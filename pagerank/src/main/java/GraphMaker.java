import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Base64;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.DataFormatException;
import java.util.zip.Inflater;

import org.apache.hadoop.util.ToolRunner;
import org.jsoup.Jsoup;
import org.jsoup.helper.Validate;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

class IntArrayWritable extends ArrayWritable {

    public IntArrayWritable(IntWritable[] values) {
        super(IntWritable.class, values);
    }

    @Override
    public IntWritable[] get() {
        return (IntWritable[]) super.get();
    }

    @Override
    public String toString() {
        IntWritable[] values = get();
        StringBuilder out = new StringBuilder();
        for (IntWritable i : values){
            out.append(i.toString());
            out.append(" ");
        }
        return out.toString().trim();
    }
}

public class GraphMaker extends Configured implements Tool {

    private static HashMap<String, Integer> index = new HashMap<>();

    static public void main(String[] args) throws Exception {
        int ret = ToolRunner.run(new GraphMaker(), args);
        System.exit(ret);
    }

    @Override
    public int run(String[] args) throws Exception {
        Job job = GetJobConf(getConf(), args[0], args[1], args[2]);
        return job.waitForCompletion(true) ? 0 : 1;
    }

    private Job GetJobConf(Configuration conf, String index_file, String input, String out_dir) throws IOException {
        Job job = Job.getInstance(conf);
        job.setJarByClass(GraphMaker.class);
        job.setJobName(GraphMaker.class.getCanonicalName());

        FileInputFormat.addInputPath(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path(out_dir));

        job.setMapperClass(ParseMapper.class);
        job.setNumReduceTasks(0);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(IntArrayWritable.class);

        Path pt=new Path(index_file);
        FileSystem fs = FileSystem.get(conf);
        BufferedReader bufferedReader =new BufferedReader(new InputStreamReader(fs.open(pt)));
        String line;
        while ((line = bufferedReader.readLine()) != null) {
            String[] split = line.split("\t");
            index.put(split[1], Integer.parseInt(split[0]));
        }

        return job;
    }

    public static class ParseMapper extends Mapper<IntWritable, Text, IntWritable, IntArrayWritable>
    {
        @Override
        protected void map(IntWritable key, Text value, Context context) throws IOException, InterruptedException {
            Inflater decompresser = new Inflater();
            Base64.Decoder decoder = Base64.getMimeDecoder();
            byte[] compressed = decoder.decode(value.toString());
            decompresser.setInput(compressed, 0, compressed.length);
            byte[] result = new byte[100*compressed.length];
            int resultLength = 0;
            try {
                resultLength = decompresser.inflate(result);
            } catch (DataFormatException e) {
                e.printStackTrace();
            }
            decompresser.end();

            String html = new String(result, 0, resultLength, "UTF-8");
            Document doc = Jsoup.parse(html);
            Elements links = doc.select("a[href]");
            HashSet<IntWritable> links_id = new HashSet<>();

            for (Element link : links) {
                String link_text = "";
                if (link.attr("abs:href").equals(""))
                    link_text = "http://lenta.ru";
                link_text += link.attr("href");
                IntWritable id = new IntWritable(index.get(link_text));
                links_id.add(id);
            }

            context.write(key, new IntArrayWritable((IntWritable[]) links_id.toArray()));
        }
    }
}
