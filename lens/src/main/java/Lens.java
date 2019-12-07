    import org.apache.lucene.analysis.TokenStream;
    import org.apache.lucene.analysis.standard.StandardAnalyzer;
    import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
    import org.apache.lucene.document.Document;
    import org.apache.lucene.document.Field;
    import org.apache.lucene.document.StringField;
    import org.apache.lucene.document.TextField;
    import org.apache.lucene.index.*;
    import org.apache.lucene.queryparser.ext.Extensions;
    import org.apache.lucene.search.*;
    import org.apache.lucene.store.Directory;
    import org.apache.lucene.store.FSDirectory;
    import org.apache.lucene.util.QueryBuilder;

    import java.io.*;
    import java.nio.file.Paths;
    import java.util.*;
    import java.util.stream.Collectors;
    import java.util.stream.IntStream;
    import java.util.stream.Stream;

    /**
     * Created by vihari on 10/08/18.
     */
    public class Lens {
	//TODO: refactor this function name
        static List<String> text(TokenStream tokenStream) throws IOException {
            CharTermAttribute charTermAttribute = tokenStream.addAttribute(CharTermAttribute.class);

            List<String> toks = new ArrayList<>();
            tokenStream.reset();
            while (tokenStream.incrementToken()) {
                String term = charTermAttribute.toString();
                if (term.length() > 2)
                    toks.add(term);
            }
            tokenStream.close();
            return toks;
        }

        static String TGT;
        static String work;
        static String SRC;
        static String src_index_dirname, t1_index_dirname;
        static float targetParam;
        static int DOCUMENT_SIZE;
	static org.apache.lucene.analysis.Analyzer analyzer = new StandardAnalyzer();

        static void set(String tgt, float t, int w) {
            SRC = "wiki";
            TGT = tgt;
            work = "vectors_" + TGT + "/ir_select";
            t1_index_dirname = work + "/t1_index_dir";
            src_index_dirname = SRC + "/index";
            targetParam = t;
            DOCUMENT_SIZE = w;
            System.out.println("TGT: " + tgt + "\n" +
                    "work dir: " + work + "\n" +
                    "target param: " + targetParam + "\n" +
                    "Document size: " + DOCUMENT_SIZE);
        }

        private static IndexWriter getWriter(Directory directory) throws IOException {
            IndexWriterConfig config = new IndexWriterConfig(analyzer);
            return new IndexWriter(directory, config);
        }

        static void WriteTgt() throws IOException {
            Runtime r = Runtime.getRuntime();
            Process p = r.exec("mkdir -p " + src_index_dirname);
	    Process p2 = r.exec("mkdir -p " + work);
            try {
                p.waitFor();
		p2.waitFor();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            if (new File(work + "/content1.txt").exists() && new File(work + "/content2.txt").exists())
                return;

            System.out.println("Splitting and writing target corpus in to two disjoints...");
            OutputStreamWriter fw1 = null, fw2 = null;
            LineNumberReader in = null;
            try {
                fw1 = new OutputStreamWriter(new FileOutputStream(work + "/content1.txt"), "UTF-8");
                fw2 = new OutputStreamWriter(new FileOutputStream(work + "/content2.txt"), "UTF-8");
                in = new LineNumberReader(new InputStreamReader(new FileInputStream("vectors_" + TGT + "/content.txt"), "UTF-8"));
            } catch (UnsupportedEncodingException | FileNotFoundException e) {
                e.printStackTrace();
            }
            String line ;
            try {
                while ((line = in.readLine()) != null) {
                    if (Math.random() < .5) fw1.write(line + "\n");
                    else fw2.write(line + "\n");
                }
                in.close();
                fw1.close();
                fw2.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
            System.out.println("Done");
        }

        public static void Index() throws IOException {
            final int SNIPPET_SIZE = DOCUMENT_SIZE;
            LineNumberReader lnrs[] = new LineNumberReader[]{
                    new LineNumberReader(new InputStreamReader(new FileInputStream("vectors_" + SRC + "/content.txt"), "UTF-8")),
                    new LineNumberReader(new InputStreamReader(new FileInputStream(work + "/content1.txt"), "UTF-8"))
            };
            String[] index_dirs = new String[]{src_index_dirname, t1_index_dirname};

            int li = 0;
            int lnri;
            for (lnri = 0; lnri < 2; lnri++) {
                if (DirectoryReader.indexExists(FSDirectory.open(Paths.get(index_dirs[lnri])))) continue;
                IndexWriter writer = getWriter(FSDirectory.open(Paths.get(index_dirs[lnri])));

                System.err.println("Indexing into " + index_dirs[lnri] + "...");
                LineNumberReader lnr = lnrs[lnri];
                List<String> bf = new ArrayList<>();
                while (true) {
                    String line = lnr.readLine();
                    if (line == null) break;
                    if (line.equals("")) {
                        String l1 = lnr.readLine();
                        if (l1 == null) break;
                        if (l1.equals("")) {
                            //new doc... reset
                            bf = new ArrayList<>();
                            continue;
                        } else lnr.setLineNumber(lnr.getLineNumber() - 1);
                    }
                    for (String w : line.split(" ")) {
                        bf.add(w);
                        if (bf.size() >= SNIPPET_SIZE) {
                            Document doc = new Document();
                            doc.add(new StringField("content", String.join(" ", bf), Field.Store.YES));
                            writer.addDocument(doc);
                            bf = new ArrayList<>();
                        }
                    }
                    li += 1;
                    if (li % 1000000 == 0)
                        System.out.print("\rProcessed lines: " + li);
                }
                System.out.println();
                lnr.close();
                writer.commit();
            }

            System.out.println("Done");
        }

	static public String getContent(String content) {
                String ct = content; //.toLowerCase();
		try{
		    return String.join(" ", text(analyzer.tokenStream("", content)).stream().collect(Collectors.toList()));
		} catch (IOException ie){
		    ie.printStackTrace();
		    return "";
		}
            }


	static class Snippet {
            public String content, example, explanation;
            public double docScore, bestScore;
            Snippet(String content, double docScore, String example, double bestScore, String explanation) {
                this.content = content;
                this.example = example;
                this.explanation = explanation;
                this.docScore = docScore;
                this.bestScore = bestScore;
            }

            public double score(){
                return this.docScore;
            }

	    public String getContent(){
		String ct = this.content.toLowerCase();
		List<String> tokens = new ArrayList<>();
                StringBuffer token = new StringBuffer();
                for (int ci=0;ci<ct.length();ci++) {
                    char c = ct.charAt(ci);
                    if (!(('a'<=c && c<='z') || ('A'<=c && c<='Z') || ('0'<=c && c<='9'))) {
                        if (c==' ')
                            tokens.add(token.toString());
                        else {
                            tokens.add(token.toString());
                            tokens.add(c + "");
                        }
                        token = new StringBuffer();
                    }
                    else token.append(c);
                }
                if (token.length()>0)
                    tokens.add(token.toString());

                return String.join(" ", tokens);

	    }
	    
            public double cooccur_score() {
                int ws = 5;
                Map<String, Map<String, Double>> vecs1 = new LinkedHashMap<>(), vecs2 = new LinkedHashMap<>();
                String[] docs = new String[]{content, example}; //.toLowerCase(), example.toLowerCase()};
                for (int k = 0; k < 2; k++) {
                    String doc = docs[k];
                    String[] text_tokens = doc.split(" ");
                    for (int i = 0; i < text_tokens.length; i++) {
                        String fw = text_tokens[i];
                        for (int j = Math.max(i - ws, 0); j < Math.min(i + ws, text_tokens.length); j++) {
                            if (j == i) continue;
                            String cw = text_tokens[j];
                            if (k == 0) {
                                Map<String, Double> _m = vecs1.getOrDefault(fw, new LinkedHashMap<>());
                                _m.put(cw, _m.getOrDefault(cw, 0.) + 1);
                                vecs1.put(fw, _m);
                            } else {
                                Map<String, Double> _m = vecs2.getOrDefault(fw, new LinkedHashMap<>());
                                _m.put(cw, _m.getOrDefault(cw, 0.) + 1);
                                vecs2.put(fw, _m);
                            }
                        }
                    }
                }
                // normalize them
                Map<String, Double> tcfreqs1 = vecs1.entrySet().stream()
                        .collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue().values().stream().reduce(Double::sum).get()));

                Map<String, Double> tcfreqs2 = vecs2.entrySet().stream()
                        .collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue().values().stream().reduce(Double::sum).get()));

		//text.toLowerCase() + " " + example.toLowerCase()
                Set<String> vocab = Stream.of((content + " " + example).split(" ")).collect(Collectors.toSet());
                double l1_dist = vocab.stream().map(s -> {
                    Map<String, Double> v1 = vecs1.getOrDefault(s, new LinkedHashMap<>());
                    Map<String, Double> v2 = vecs2.getOrDefault(s, new LinkedHashMap<>());
                    Double f1 = tcfreqs1.get(s);
                    Double f2 = tcfreqs2.get(s);
                    if (f1==null) f1 = 1.;
                    if (f2==null) f2 = 1.;
                    try {
                        Double finalF = f1;
                        Double finalF1 = f2;
                        return vocab.stream().map(s1 -> Math.abs(v1.getOrDefault(s1, 0.) / finalF - v2.getOrDefault(s1, 0.) / finalF1)).reduce(Double::sum).get();
                    } catch(NullPointerException ne){
                        ne.printStackTrace();
                        System.err.println("v1: " + v1 + " v2: " + v2 + " f1: " + f1 + " f2: " + f2 + " s: " + s + " text:" + content + " example: " + example);
                        return Double.MAX_VALUE;
                    }
                }).reduce(Double::sum).get();
                return l1_dist;
            }
        }

        static void run() throws IOException {
            WriteTgt();
            Directory src_index_dir, t1_index_dir;
            IndexReader _r1, _r2;
            try {
                src_index_dir = FSDirectory.open(Paths.get(src_index_dirname));
                t1_index_dir = FSDirectory.open(Paths.get(t1_index_dirname));
                _r1 = DirectoryReader.open(src_index_dir);
                _r2 = DirectoryReader.open(t1_index_dir);
            } catch (IndexNotFoundException infe) {
                System.out.println("No index found, creating one...");
                Index();
                src_index_dir = FSDirectory.open(Paths.get(src_index_dirname));
                t1_index_dir = FSDirectory.open(Paths.get(t1_index_dirname));
                _r1 = DirectoryReader.open(src_index_dir);
                _r2 = DirectoryReader.open(t1_index_dir);
            }
            final IndexReader src_index_reader = _r1, t1_index_reader = _r2;

 	    //new StandardAnalyzer();
            LineNumberReader[] lnrs = new LineNumberReader[]{new LineNumberReader(new FileReader("vectors_" + SRC + "/content.txt")), new LineNumberReader(new FileReader("vectors_" + TGT + "/content.txt"))};
            long _d = 0;
            Map<String, Integer> freqs = new LinkedHashMap<>();
            Map<String, Integer> tgtFreqs = new LinkedHashMap<>();

            for (int i = 1; i < lnrs.length; i++) {
                LineNumberReader lnr1 = lnrs[i];
                String line;
                while ((line = lnr1.readLine()) != null) {
                    Set<String> words = text(analyzer.tokenStream("", line)).stream().collect(Collectors.toSet());
                    for (String w : words) {
                        freqs.put(w, freqs.getOrDefault(w, 0) + 1);
                        if (i == 1) tgtFreqs.put(w, tgtFreqs.getOrDefault(w, 0) + 1);
                        _d++;
                    }
                }
            }

            Map<String, Double> idfs = new LinkedHashMap<>();
            for (String str : freqs.keySet()) {
                // double _v = (double)_d/(1+tgtFreqs.getOrDefault(str, 0));
                //_v *= Math.pow(100 + tgtFreqs.getOrDefault(str, 0), .75);
                double _v = Math.log(((double) _d) / (1 + tgtFreqs.getOrDefault(str, 0)));
                idfs.put(str, _v);
            }

            LineNumberReader lnr = new LineNumberReader(new FileReader(work + "/content2.txt"));
            int t2_ws = DOCUMENT_SIZE;
            int t2_docid = 0;
            Map<String, Map<Integer, Double>> termScores = new LinkedHashMap<>();
            List<String> lines = lnr.lines().collect(Collectors.toList());
            List<String> t2s = new ArrayList<>();
            List<String> bf = new ArrayList<>();
            for (String l : lines) {
                List<String> toks = null;
                try {
                    toks = text(analyzer.tokenStream("", l)).stream().collect(Collectors.toList());
                } catch (IOException e) {
                    e.printStackTrace();
                }
                for (String tok : toks) {
                    bf.add(tok);
                    if (bf.size() >= t2_ws) {
                        double _s = 0;
                        Map<String, Double> l_termscores = new LinkedHashMap<>();
                        List<String> _bf = new ArrayList<>();
                        for (String w : bf) {
                            if (_bf.contains(w)) continue;
                            _bf.add(w);

                            _s += idfs.get(w);
                            l_termscores.put(w, l_termscores.getOrDefault(w, 0.) + idfs.get(w));
                        }
                        // to fight noise
                        if (bf.stream().collect(Collectors.toSet()).size() < (t2_ws / 2)) {
                            bf = new ArrayList<>();
                            continue;
                        }
                        for (Map.Entry<String, Double> e : l_termscores.entrySet()) {
                            Map<Integer, Double> _m = termScores.getOrDefault(e.getKey(), new LinkedHashMap<>());
                            double _v = e.getValue() / _s;
                            if (_v > 0.0) {
                                _m.put(t2_docid, _v);
                                termScores.put(e.getKey(), _m);
                            }
                        }
                        t2s.add(String.join(" ", bf));
                        t2_docid++;
                        bf = new ArrayList<>();
                    }
                }
            }

            float lo = 0, hi = 1.0f;
            int num_probes = 0;
            float target = targetParam;
            double docThreshold;

            while (true) {
                Map<Integer, Double> docScores = new LinkedHashMap<>();

                System.out.println("Num src docs: " + src_index_reader.numDocs());
                System.out.println("Num t1 docs: " + t1_index_reader.numDocs());

                float eps = (lo + hi) / 2;
                System.out.println("Threshold: " + eps);
                IntStream.range(0, t1_index_reader.maxDoc()).parallel()
                        .map(sdoc -> {
                            try {
                                Document doc = t1_index_reader.document(sdoc);
                                Set<String> toks = text(analyzer.tokenStream("", doc.get("content"))).stream().collect(Collectors.toSet());
                                Map<Integer, Double> agg_map = new LinkedHashMap<>();
                                for (String tok : toks)
                                    if (termScores.containsKey(tok))
                                        for (Map.Entry<Integer, Double> e : termScores.get(tok).entrySet()) {
                                            double _v = agg_map.getOrDefault(e.getKey(), 0.) + e.getValue();
                                            agg_map.put(e.getKey(), _v);
                                        }

                                Double agg_val = agg_map.values().stream()
                                        .filter(s -> s > eps)
                                        .reduce(Double::sum)
                                        .orElseGet(() -> 0.);
                                if (agg_val > eps)
                                    docScores.put(sdoc, agg_val);
                                //if (docScores.size() % 1000 == 0)
                                System.err.print("\r" + docScores.size() + "/" + t1_index_reader.numDocs());
                            } catch (IOException ie) {
                                ie.printStackTrace();
                            }
                            return -1;
                        }).reduce(Integer::max);

                float fraction = ((float) docScores.size()) / t1_index_reader.numDocs();
                //if ((hi - lo) < 0.1) break;
                if (num_probes >= 10 || Math.abs(fraction - target) < 0.01) {
                    System.out.println("Size: " + docScores.size());
                    List<Double> sortedDocScores = docScores.values().stream().collect(Collectors.toList());
                    Collections.sort(sortedDocScores);
                    docThreshold = sortedDocScores.get((int) ((1 - target) * sortedDocScores.size()));
                    System.err.println("Num probes: " + num_probes + " -- " + fraction);

                    break;
                }
                if (fraction < target) hi = (lo + hi) / 2;
                else lo = (hi + lo) / 2;
                num_probes++;
                System.err.println("Num probes: " + num_probes + " -- " + fraction);
            }
            System.out.println("Getting...");
            double queryThresh = (lo + hi) / 2;

            System.err.println("Threshold: " + queryThresh + " Doc thresh: " + docThreshold);

            OutputStreamWriter osw = new OutputStreamWriter(new FileOutputStream(work + "/selected.txt"));
            OutputStreamWriter ssw = new OutputStreamWriter(new FileOutputStream(work + "/ir-doc-scores.txt"));
            OutputStreamWriter osw2 = new OutputStreamWriter(new FileOutputStream(work + "/selected-debug.txt"));
            List<Integer> arr = new ArrayList<>();
            List<Snippet> snippets = new ArrayList<>();
            List<Snippet> finalSnippets = snippets;
            IntStream.range(0, src_index_reader.numDocs())
		    .parallel()
                    .map(i -> {
                        try {
                            Document doc = src_index_reader.document(i);

			    String original_content = doc.get("content");
			    long brac_count = original_content.chars().filter(x-> x=='(').count();
			    // if more than 3 round brackets, somkind of a list 
			    if (brac_count >= 3) return 0;
			    
                            String content = getContent(original_content);
                            List<String> toks = Arrays.asList(content.split(" "));
                            Map<Integer, Double> agg_map = new LinkedHashMap<>();
                            toks.stream().filter(termScores::containsKey).forEach(w -> {
                                for (Map.Entry<Integer, Double> e : termScores.get(w).entrySet())
                                    agg_map.put(e.getKey(), agg_map.getOrDefault(e.getKey(), 0.) + e.getValue());
                            });

                            Map.Entry<Integer, Double> best_val = agg_map.entrySet().stream()
                                    .reduce((a, b) -> a.getValue() > b.getValue() ? a : b).orElseGet(() -> null);
                            double docScore = agg_map.values().stream().filter(s -> s > queryThresh).reduce(Double::sum).orElse(0.);
                            if (best_val != null && docScore > docThreshold) {
                                String[] _ws = t2s.get(best_val.getKey()).split(" ");
                                String explanation = String.join(" -- ", Stream.of(_ws).filter(toks::contains).map(s -> s + "(" + termScores.get(s).get(best_val.getKey()) + ")").collect(Collectors.toList()));
                                finalSnippets.add(new Snippet(original_content, docScore, t2s.get(best_val.getKey()), best_val.getValue(), explanation));
                            }
                            // randomly add 5% of the rest
                            else if (Math.random()<=0.05)
                                finalSnippets.add(new Snippet(original_content, 0, "", 0, ""));

                            //just for progress tracking
                            arr.add(0);
                            if (arr.size() % 1000 == 0)
                                System.err.print("\r" + arr.size() + " of 6M");
                        } catch (IOException ie) {
                            ie.printStackTrace();
                        }
                        return 0;
                    }).reduce(Integer::max);

            System.out.println("Number of null docs: " + snippets.stream().filter(a->a==null).count() + " Total: " + snippets.size()) ;
            snippets = snippets.stream().filter(a->a!=null).sorted((a,b)->Double.compare(-a.score(), -b.score())).collect(Collectors.toList());
            for(Snippet sn: snippets) {
                osw.write(sn.getContent() + "\n");
                ssw.write(String.format("%.3f", sn.docScore) + "\n");
                if (sn.docScore==0) continue;

                //":::" + String.format("%.3f", sn.docScore) + ":::" + String.format("%.3f", sn.bestScore) + "\n");
                String str = "";
                str += sn.getContent() + "\nDoc score: " + String.format("%.3f", sn.docScore) + " -- Best score: " + String.format("%.3f", sn.bestScore) + "\n";
                //str += "CScore: " + String.format("%.3f", sn.cooccur_score());
                str += "-----------------\n";
                str += sn.example + "\n";
                str += "-----------------\n";

                str += sn.explanation+"\n";
                osw2.write(str + "\n");
            }
            osw.close();
            ssw.close();
            osw2.close();
        }

        public static void main(String[] args) throws IOException {
	    /**
	       Expects three command line arguments.
	       1. The name of your target, for example "physics" when you have a folder named `vector_physics` in the base folder with 'content.txt' in it.
	       2. Coverage hyperparameter, a float value between 0 and 1. This is expected in-domain coverage fraction. Should be high ~0.9, but not 1 to overfit.
	       3. Window size, natural number. The content in source is split in to multiple mini-documents of this size (words) -- quantum of selection.
	    */
            set(args[0], Float.parseFloat(args[1]), Integer.parseInt(args[2]));
            //WriteTgt();
            run();
        }
    }
