package mahouttest.test;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.clustering.kmeans.Cluster;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.common.AbstractJob;
//import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;


public final class TestKmeansClustering {

	public static final double[][] REFERENCE = { { 1, 1, 0 }, { 1, 2, 1 },
			{ 2, 2, 10 }, { 3, 3, 11 }, { 4, 4, 5 }, { 5, 4,2 }, { 4, 5, 3 }, { 5, 5, 5 }, { 1, 2, 0 },  };


	public static void main(String[] args) throws Exception {
		//List<VectorWritable> points = getPointsWritable(REFERENCE);
		List<VectorWritable> points = getPointsFromFile
			("/home/haruyama/kakakucate.csv");
		
		if (args.length > 0) {
			testKMeansSeqJob(points, Integer.parseInt(args[0]));
		} else {
			testKMeansSeqJob(points, 10);
		}
	}

	public static List<VectorWritable> getPointsWritable(double[][] raw) {
		List<VectorWritable> points = new ArrayList<VectorWritable>();
		for (double[] fr : raw) {
			Vector vec = new RandomAccessSparseVector(fr.length);
			vec.assign(fr);
			points.add(new VectorWritable(vec));
		}
		return points;
	}
	
	private static double[] line2DoubleArray(String line) {
		StringTokenizer st = new StringTokenizer(line, ",");
		List<Double> list = new ArrayList<Double>();
		boolean isFirst = true; 
		while (st.hasMoreElements()) {
			if (isFirst) {
				isFirst = false;
				st.nextToken();
			} else {
				list.add(Double.parseDouble(st.nextToken()));
			}
		}
		double[] ret = new double[list.size()];
		int i = 0;
		for (double d : list) {
			ret[i++] = d;
		}
		//System.out.println(ret);
		return ret;
	}
	
	private static List<VectorWritable> getPointsFromFile(String filename) {
		BufferedReader reader = null;
		List<VectorWritable> points = new ArrayList<VectorWritable>();
		try {
			reader = new BufferedReader(new FileReader(filename));
			String line = null;
			while((line = reader.readLine()) != null) {
				double[] da = line2DoubleArray(line);
				Vector vec = new RandomAccessSparseVector(da.length);
				vec.assign(da);
				points.add(new VectorWritable(vec));
			}
		} catch (Exception ex) {
			ex.printStackTrace();
			System.exit(1);

		} finally {
			try {
				if (reader !=null) reader.close();
			} catch (Exception e) {
				e.printStackTrace();
				System.exit(1);
			}
		}
		return points;
		
	}


	private static void testKMeansSeqJob(
			List<VectorWritable> points, int k) throws Exception {
		DistanceMeasure measure = new EuclideanDistanceMeasure();

		Path pointsPath = new Path("points");
		Path clustersPath = new Path("clusters");
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		writePointsToFile(points, new Path(pointsPath, "file1"), fs, conf);
		System.err.println("testKMeansMRJob k= " + k);
		// now run the Job
		Path outputPath = new Path("output");
		fs.delete(outputPath, true);

		/*
		//CanopyDriver.run(pointsPath, outputPath, new ManhattanDistanceMeasure(), 3.0, 2.0, false, false);
		CanopyDriver.run(pointsPath, outputPath, new EuclideanDistanceMeasure(), 2.2, 1.0, false, false);

	    // now run the KMeans job
	    KMeansDriver.run(pointsPath,
	    		new Path(outputPath, "clusters-0"),
	    		outputPath,
	    		new EuclideanDistanceMeasure(),
	    		0.001,
	    		100,
	    		true,
	    		false);
	    Path clusteredPointsPath = new Path(outputPath, "clusteredPoints");
		SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(
				clusteredPointsPath, "part-m-00000"), conf);


		*/
		
		
		// pick k initial cluster centers at random
		Path path = new Path(clustersPath, "part-00000");
		
		SequenceFile.Writer writer = new SequenceFile.Writer(FileSystem.get(
				path.toUri(), conf), conf, path, Text.class, Cluster.class);

		for (int i = 0; i < k; ++i) {
			Vector vec = points.get(i).get();
			Cluster cluster = new Cluster(vec, i, measure);
			// add the center so the centroid will be correct upon output
			cluster.observe(cluster.getCenter(), 1);
			writer.append(new Text(cluster.getIdentifier()), cluster);
		}
		writer.close();

		String[] args = { optKey(DefaultOptionCreator.INPUT_OPTION),
				pointsPath.toString(),
				optKey(DefaultOptionCreator.CLUSTERS_IN_OPTION),
				clustersPath.toString(),
				optKey(DefaultOptionCreator.OUTPUT_OPTION),
				outputPath.toString(),
				optKey(DefaultOptionCreator.DISTANCE_MEASURE_OPTION),
				EuclideanDistanceMeasure.class.getName(),
				optKey(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION), "0.001",
				optKey(DefaultOptionCreator.MAX_ITERATIONS_OPTION), "100",
				optKey(DefaultOptionCreator.CLUSTERING_OPTION),
				optKey(DefaultOptionCreator.OVERWRITE_OPTION),
				optKey(DefaultOptionCreator.METHOD_OPTION),
				DefaultOptionCreator.SEQUENTIAL_METHOD };
		new KMeansDriver().run(args);
		//KMeansDriver.run(pointsPath, clustersPath, outputPath, new EuclideanDistanceMeasure(), 0.001, 100, true, false);
		Path clusteredPointsPath = new Path(outputPath, "clusteredPoints");
		SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(
				clusteredPointsPath, "part-m-0"), conf);

		
		// now compare the expected clusters with actual
		IntWritable clusterId = new IntWritable(0);
		// The value is the weighted vector
		WeightedVectorWritable value = new WeightedVectorWritable();
		while (reader.next(clusterId, value)) {
			System.out.println(clusterId.toString());
			//System.out.println(value.toString());
			//System.out.println();
		}
		reader.close();
	}

	public static void writePointsToFile(Iterable<VectorWritable> points,
			Path path, FileSystem fs, Configuration conf) throws IOException {
		writePointsToFile(points, false, path, fs, conf);
	}

	public static void writePointsToFile(Iterable<VectorWritable> points,
			boolean intWritable, Path path, FileSystem fs, Configuration conf)
			throws IOException {
		SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path,
				intWritable ? IntWritable.class : LongWritable.class,
				VectorWritable.class);
		int recNum = 0;
		for (VectorWritable point : points) {
			writer.append(intWritable ? new IntWritable(recNum++)
					: new LongWritable(recNum++), point);
		}
		writer.close();
	}

	protected static String optKey(String optionName) {
		return AbstractJob.keyFor(optionName);
	}

	
}
