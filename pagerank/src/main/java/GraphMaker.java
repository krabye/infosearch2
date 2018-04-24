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
import java.util.zip.DataFormatException;
import java.util.zip.Inflater;

import org.apache.hadoop.util.ToolRunner;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

class LongArrayWritable extends ArrayWritable {

    LongArrayWritable(LongWritable[] values) {
        super(LongWritable.class, values);
    }

    @Override
    public LongWritable[] get() {
        return (LongWritable[]) super.get();
    }

    @Override
    public String toString() {
        LongWritable[] values = get();
        StringBuilder out = new StringBuilder();
        for (LongWritable i : values){
            out.append(i.toString());
            out.append(" ");
        }
        return out.toString().trim();
    }
}

public class GraphMaker extends Configured implements Tool {

    static public void main(String[] args) throws Exception {
        int ret = ToolRunner.run(new GraphMaker(), args);
        System.exit(ret);
    }

    @Override
    public int run(String[] args) throws Exception {
        Job job = GetJobConf(getConf(), args[0], args[1], args[2]);
        return job.waitForCompletion(true) ? 0 : 1;
    }

    private Job GetJobConf(Configuration conf, String index_file_name, String input, String out_dir) throws IOException {
        conf.set("index_file", index_file_name);
        System.out.println("Index name :" + conf.get("index_file"));

        Job job = Job.getInstance(conf);
        job.setJarByClass(GraphMaker.class);
        job.setJobName(GraphMaker.class.getCanonicalName());

        FileInputFormat.addInputPath(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path(out_dir));

        job.setMapperClass(ParseMapper.class);
        job.setNumReduceTasks(0);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(LongArrayWritable.class);

        return job;
    }

    public static class ParseMapper extends Mapper<LongWritable, Text, LongWritable, LongArrayWritable>
    {
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            HashMap<String, Long> index = new HashMap<>();
            Configuration conf = context.getConfiguration();
            System.out.println(conf.get("index_file"));
            Path pt=new Path(conf.get("index_file"));
            FileSystem fs = FileSystem.get(conf);
            BufferedReader bufferedReader =new BufferedReader(new InputStreamReader(fs.open(pt)));
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                String[] split = line.split("\t");
                index.put(split[1], Long.parseLong(split[0]));
            }

            String[] split = value.toString().split("\t");
            LongWritable idx = new LongWritable(Long.parseLong(split[0]));
            String base64 = split[1];
            Inflater decompresser = new Inflater();
            Base64.Decoder decoder = Base64.getMimeDecoder();
            byte[] compressed = decoder.decode(base64);
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
            HashSet<LongWritable> links_id = new HashSet<>();

            System.out.println("Len of map:");
            System.out.println(index.size());

            for (Element link : links) {
                String link_text = "";
                if (link.attr("abs:href").equals(""))
                    link_text = "http://lenta.ru";
                link_text += link.attr("href");
                LongWritable id = new LongWritable(index.get(link_text));
                links_id.add(id);
            }

            context.write(idx, new LongArrayWritable((LongWritable[]) links_id.toArray()));
        }
    }
}
