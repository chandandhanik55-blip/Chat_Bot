import com.sun.net.httpserver.HttpServer;
import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.io.*;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.LocalDate;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.*;
import java.util.regex.Pattern;

/**
 * FAQ ChatBot - single file, no external dependencies (Java 11+).
 *
 * Run GUI : javac ChatBot.java && java ChatBot
 * Run web : java ChatBot web      (open http://localhost:8080)
 * Console : java ChatBot console
 *
 * NLP pipeline : lowercase -> tokenize -> stop-word removal -> light stemming
 * ML logic     : TF-IDF vectors + cosine similarity (nearest FAQ question)
 * Rule-based   : regex intents (greeting, thanks, bye, help, time, date)
 * Training     : edit faq.txt (question|answer) or type  "teach: question => answer"
 */
public class ChatBot {

    // ======================= NLP =======================
    static class NLP {
        static final Set<String> STOP = new HashSet<>(Arrays.asList(
            "a","an","the","is","are","was","were","be","am","do","does","did","to","of","in","on",
            "for","and","or","it","its","i","me","my","you","your","we","our","can","could","would",
            "should","will","this","that","with","at","by","from","as","there","please","tell","about",
            "what","how","who","when","where","which","why","if","so","just","any","some"));

        static List<String> tokenize(String text) {
            List<String> out = new ArrayList<>();
            for (String w : text.toLowerCase().replaceAll("[^a-z0-9\\s]", " ").split("\\s+")) {
                if (w.isEmpty() || STOP.contains(w)) continue;
                out.add(stem(w));
            }
            return out;
        }

        // Lightweight suffix-stripping stemmer
        static String stem(String w) {
            if (w.length() <= 3) return w;
            String[] suffixes = {"ingly","edly","ing","ies","ied","ed","es","ly","s"};
            for (String s : suffixes) {
                if (w.endsWith(s) && w.length() - s.length() >= 3) {
                    String base = w.substring(0, w.length() - s.length());
                    return (s.equals("ies") || s.equals("ied")) ? base + "y" : base;
                }
            }
            return w;
        }
    }

    // ================= ML: TF-IDF FAQ model =================
    static class FaqModel {
        static class Entry {
            String question, answer;
            Map<String, Double> vec = new HashMap<>();
            double norm;
            Entry(String q, String a) { question = q; answer = a; }
        }

        final List<Entry> entries = new ArrayList<>();
        final Map<String, Double> idf = new HashMap<>();
        static final double THRESHOLD = 0.25;

        void add(String q, String a) { entries.add(new Entry(q, a)); }

        /** (Re)train: compute IDF and a TF-IDF vector for every FAQ question. */
        void train() {
            idf.clear();
            Map<String, Integer> df = new HashMap<>();
            List<List<String>> docs = new ArrayList<>();
            for (Entry e : entries) {
                // include answer text at lower weight by simply adding its tokens once
                List<String> toks = NLP.tokenize(e.question);
                docs.add(toks);
                for (String t : new HashSet<>(toks)) df.merge(t, 1, Integer::sum);
            }
            int n = entries.size();
            for (Map.Entry<String, Integer> d : df.entrySet())
                idf.put(d.getKey(), Math.log((n + 1.0) / (d.getValue() + 1.0)) + 1.0);

            for (int i = 0; i < n; i++) {
                Entry e = entries.get(i);
                e.vec = vectorize(docs.get(i));
                e.norm = norm(e.vec);
            }
        }

        Map<String, Double> vectorize(List<String> tokens) {
            Map<String, Double> tf = new HashMap<>();
            for (String t : tokens) tf.merge(t, 1.0, Double::sum);
            Map<String, Double> v = new HashMap<>();
            for (Map.Entry<String, Double> e : tf.entrySet())
                v.put(e.getKey(), e.getValue() * idf.getOrDefault(e.getKey(), 1.0));
            return v;
        }

        static double norm(Map<String, Double> v) {
            double s = 0;
            for (double x : v.values()) s += x * x;
            return Math.sqrt(s);
        }

        /** Returns {entry, score} for the best match, or null if no entries. */
        Object[] bestMatch(String query) {
            Map<String, Double> q = vectorize(NLP.tokenize(query));
            double qn = norm(q);
            if (qn == 0) return null;
            Entry best = null;
            double bestScore = 0;
            for (Entry e : entries) {
                if (e.norm == 0) continue;
                double dot = 0;
                for (Map.Entry<String, Double> t : q.entrySet()) {
                    Double w = e.vec.get(t.getKey());
                    if (w != null) dot += t.getValue() * w;
                }
                double score = dot / (qn * e.norm);
                if (score > bestScore) { bestScore = score; best = e; }
            }
            return best == null ? null : new Object[]{best, bestScore};
        }
    }

    // ================= Rule-based intents =================
    static class Intent {
        final Pattern pattern;
        final java.util.function.Supplier<String> reply;
        Intent(String regex, java.util.function.Supplier<String> reply) {
            this.pattern = Pattern.compile(regex, Pattern.CASE_INSENSITIVE);
            this.reply = reply;
        }
    }

    // ================= Bot engine =================
    static class Bot {
        final Path faqFile = Paths.get("faq.txt");
        final FaqModel model = new FaqModel();
        final List<Intent> intents = new ArrayList<>();
        final Random rnd = new Random();

        Bot() {
            intents.add(new Intent("^(hi|hello|hey|good (morning|afternoon|evening)|greetings)\\b.*",
                () -> pick("Hello! How can I help you today?", "Hi there! Ask me anything.", "Hey! What can I do for you?")));
            intents.add(new Intent(".*\\b(thanks|thank you|thx|cheers)\\b.*",
                () -> pick("You're welcome!", "Happy to help!", "Anytime!")));
            intents.add(new Intent("^(bye|goodbye|see you|exit|quit)\\b.*",
                () -> pick("Goodbye! Have a great day.", "See you soon!")));
            intents.add(new Intent(".*\\b(what time|current time|time is it)\\b.*",
                () -> "It's " + LocalTime.now().format(DateTimeFormatter.ofPattern("hh:mm a")) + "."));
            intents.add(new Intent(".*\\b(what('s| is) (the )?date|today's date|what day)\\b.*",
                () -> "Today is " + LocalDate.now().format(DateTimeFormatter.ofPattern("EEEE, MMMM d, yyyy")) + "."));
            intents.add(new Intent("^(help|\\?|what can you do).*",
                () -> "Ask me a question about our products or support. You can also teach me:\n"
                    + "  teach: your question => the answer"));
            loadFaq();
        }

        String pick(String... options) { return options[rnd.nextInt(options.length)]; }

        void loadFaq() {
            try {
                if (!Files.exists(faqFile)) Files.write(faqFile, defaultFaq().getBytes(StandardCharsets.UTF_8));
                for (String line : Files.readAllLines(faqFile, StandardCharsets.UTF_8)) {
                    line = line.trim();
                    if (line.isEmpty() || line.startsWith("#")) continue;
                    int i = line.indexOf('|');
                    if (i > 0) model.add(line.substring(0, i).trim(), line.substring(i + 1).trim());
                }
            } catch (IOException e) {
                for (String line : defaultFaq().split("\n")) {
                    int i = line.indexOf('|');
                    if (i > 0 && !line.startsWith("#")) model.add(line.substring(0, i).trim(), line.substring(i + 1).trim());
                }
            }
            model.train();
        }

        static String defaultFaq() {
            return "# question|answer  (one per line, edit freely and restart)\n"
                + "What are your opening hours?|We're open Monday to Friday, 9 AM to 6 PM.\n"
                + "How can I reset my password?|Click 'Forgot password' on the login page and follow the emailed link.\n"
                + "How do I contact customer support?|Email support@example.com or call 1-800-555-0199.\n"
                + "What is your refund policy?|You can request a full refund within 30 days of purchase.\n"
                + "Do you offer free shipping?|Yes, free shipping on orders over $50.\n"
                + "How long does delivery take?|Standard delivery takes 3-5 business days.\n"
                + "How can I track my order?|Use the tracking link in your confirmation email or visit Orders in your account.\n"
                + "What payment methods do you accept?|We accept credit cards, PayPal and bank transfer.\n"
                + "How do I cancel my subscription?|Go to Account > Subscription > Cancel. You keep access until the period ends.\n"
                + "Where is your office located?|Our head office is at 123 Main Street, Springfield.\n"
                + "Do you have a mobile app?|Yes, our app is available on iOS and Android.\n"
                + "How do I create an account?|Click 'Sign up', enter your email and choose a password.\n";
        }

        String reply(String input) {
            String text = input == null ? "" : input.trim();
            if (text.isEmpty()) return "Please type something.";

            // 1. Teaching command (online learning)
            if (text.toLowerCase().startsWith("teach:")) return teach(text.substring(6));

            // 2. Rule-based intents
            for (Intent in : intents) if (in.pattern.matcher(text).matches()) return in.reply.get();

            // 3. ML: nearest FAQ by TF-IDF cosine similarity
            Object[] m = model.bestMatch(text);
            if (m != null) {
                FaqModel.Entry e = (FaqModel.Entry) m[0];
                double score = (double) m[1];
                if (score >= FaqModel.THRESHOLD) return e.answer;
                if (score >= 0.12) return "I'm not sure, but did you mean \"" + e.question + "\"?\n" + e.answer;
            }
            return "Sorry, I don't know that yet. You can teach me with:\n  teach: your question => the answer";
        }

        String teach(String body) {
            int i = body.indexOf("=>");
            if (i < 0) return "Format: teach: question => answer";
            String q = body.substring(0, i).trim(), a = body.substring(i + 2).trim();
            if (q.isEmpty() || a.isEmpty()) return "Format: teach: question => answer";
            model.add(q, a);
            model.train();
            try {
                Files.write(faqFile, ("\n" + q.replace("|", "/") + "|" + a.replace("\n", " ")).getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.APPEND, StandardOpenOption.CREATE);
            } catch (IOException ignored) { }
            return "Got it! I've learned that.";
        }
    }

    // ================= Swing GUI =================
    static void launchGui(Bot bot) {
        SwingUtilities.invokeLater(() -> {
            JFrame f = new JFrame("FAQ ChatBot");
            f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            f.setSize(520, 620);
            f.setLocationRelativeTo(null);

            JTextArea chat = new JTextArea();
            chat.setEditable(false);
            chat.setLineWrap(true);
            chat.setWrapStyleWord(true);
            chat.setFont(new Font("SansSerif", Font.PLAIN, 14));
            chat.setMargin(new Insets(10, 10, 10, 10));
            chat.append("Bot: Hi! Ask me a question (type 'help' for tips).\n\n");

            JTextField input = new JTextField();
            input.setFont(new Font("SansSerif", Font.PLAIN, 14));
            JButton send = new JButton("Send");

            Runnable submit = () -> {
                String msg = input.getText().trim();
                if (msg.isEmpty()) return;
                input.setText("");
                chat.append("You: " + msg + "\n");
                // small delay for a "typing" feel, off the EDT
                new SwingWorker<String, Void>() {
                    protected String doInBackground() throws Exception {
                        Thread.sleep(350);
                        return bot.reply(msg);
                    }
                    protected void done() {
                        try { chat.append("Bot: " + get() + "\n\n"); } catch (Exception ex) { chat.append("Bot: error\n\n"); }
                        chat.setCaretPosition(chat.getDocument().getLength());
                    }
                }.execute();
            };
            send.addActionListener(e -> submit.run());
            input.addKeyListener(new KeyAdapter() {
                public void keyPressed(KeyEvent e) { if (e.getKeyCode() == KeyEvent.VK_ENTER) submit.run(); }
            });

            JPanel bottom = new JPanel(new BorderLayout(6, 6));
            bottom.setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));
            bottom.add(input, BorderLayout.CENTER);
            bottom.add(send, BorderLayout.EAST);

            f.add(new JScrollPane(chat), BorderLayout.CENTER);
            f.add(bottom, BorderLayout.SOUTH);
            f.setVisible(true);
            input.requestFocusInWindow();
        });
    }

    // ================= Web interface =================
    static final String PAGE = "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        + "<meta name='viewport' content='width=device-width,initial-scale=1'><title>FAQ ChatBot</title>"
        + "<style>body{font-family:system-ui,sans-serif;background:#f0f2f5;margin:0;display:flex;justify-content:center}"
        + "#box{width:100%;max-width:520px;height:100vh;display:flex;flex-direction:column;background:#fff}"
        + "h1{margin:0;padding:14px;font-size:18px;background:#2563eb;color:#fff}"
        + "#log{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:8px}"
        + ".m{max-width:80%;padding:8px 12px;border-radius:14px;white-space:pre-wrap;line-height:1.35}"
        + ".u{align-self:flex-end;background:#2563eb;color:#fff}.b{align-self:flex-start;background:#e5e7eb}"
        + "form{display:flex;gap:8px;padding:10px;border-top:1px solid #ddd}"
        + "input{flex:1;padding:10px;border:1px solid #ccc;border-radius:8px;font-size:15px}"
        + "button{padding:10px 16px;border:0;border-radius:8px;background:#2563eb;color:#fff;font-size:15px}</style></head>"
        + "<body><div id='box'><h1>FAQ ChatBot</h1><div id='log'></div>"
        + "<form id='f'><input id='q' autocomplete='off' placeholder='Ask a question...' autofocus><button>Send</button></form></div>"
        + "<script>const log=document.getElementById('log');"
        + "function add(t,c){const d=document.createElement('div');d.className='m '+c;d.textContent=t;log.appendChild(d);log.scrollTop=log.scrollHeight}"
        + "add('Hi! Ask me a question (type help for tips).','b');"
        + "document.getElementById('f').onsubmit=async e=>{e.preventDefault();const i=document.getElementById('q');const v=i.value.trim();if(!v)return;"
        + "i.value='';add(v,'u');try{const r=await fetch('/chat',{method:'POST',body:v});add(await r.text(),'b')}catch(x){add('Connection error','b')}}"
        + "</script></body></html>";

    static void launchWeb(Bot bot, int port) throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext("/", ex -> {
            byte[] body = PAGE.getBytes(StandardCharsets.UTF_8);
            ex.getResponseHeaders().add("Content-Type", "text/html; charset=utf-8");
            ex.sendResponseHeaders(200, body.length);
            try (OutputStream os = ex.getResponseBody()) { os.write(body); }
        });
        server.createContext("/chat", ex -> {
            String in = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            String out;
            synchronized (bot) { out = bot.reply(in); }
            byte[] body = out.getBytes(StandardCharsets.UTF_8);
            ex.getResponseHeaders().add("Content-Type", "text/plain; charset=utf-8");
            ex.sendResponseHeaders(200, body.length);
            try (OutputStream os = ex.getResponseBody()) { os.write(body); }
        });
        server.start();
        System.out.println("Web chat running at http://localhost:" + port);
    }

    // ================= Main =================
    public static void main(String[] args) throws Exception {
        Bot bot = new Bot();
        String mode = args.length > 0 ? args[0].toLowerCase() : "gui";
        switch (mode) {
            case "web":
                launchWeb(bot, 8080);
                break;
            case "console":
                BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
                System.out.println("Bot: Hi! Type 'quit' to exit.");
                String line;
                while ((line = br.readLine()) != null) {
                    String r = bot.reply(line);
                    System.out.println("Bot: " + r);
                    if (line.trim().matches("(?i)^(bye|goodbye|quit|exit)\\b.*")) break;
                }
                break;
            default:
                launchGui(bot);
        }
    }
}