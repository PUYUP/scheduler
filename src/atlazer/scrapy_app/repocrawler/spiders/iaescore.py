import structlog
import scrapy

log = structlog.get_logger(__name__)


class IaeScoreSpider(scrapy.Spider):
    name = "iaescore"
    
    def __init__(self, journal, volume, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.journal = journal
        self.volume = volume
        self.start_urls = [f"https://{journal}.iaescore.com/index.php/{journal.capitalize()}/issue/archive"]

        log.info(
            "spider_initialized",
            journal=self.journal,
            volume=self.volume,
            start_urls=self.start_urls,
        )

    def parse(self, response):
        issues = response.css('div[id^="issue-"]')
        log.info("issues_found", count=len(issues), url=response.url)

        for issue in issues:
            issue_id = issue.attrib.get("id")           # "issue-588"
            issue_number = int(issue_id.split("-")[1])  # 588
            link = issue.css("h4 a::attr(href)").get()
            title = issue.css("h4 a::text").get()

            # patokan langsung ke issue_number, bukan parsing "Vol X" dari title
            if issue_number == int(self.volume):
                log.info("issue_matched", issue_number=issue_number, title=title, url=link)
                yield response.follow(
                    link,
                    callback=self.parse_issue,
                    cb_kwargs={"issue_number": issue_number, "issue_title": title},
                )
            else:
                log.debug("issue_skipped", issue_number=issue_number, title=title)

        # lanjut ke halaman pagination berikutnya
        # next_page = response.css('a[href*="issuesPage"]::attr(href)').getall()
        # if next_page:
        #     # ambil link "next" terakhir yang unik (hindari duplikasi karena ada link '>' dan '>>')
        #     for np in next_page:
        #         yield response.follow(np, callback=self.parse)

    def parse_issue(self, response, issue_number, issue_title):
        log.info("parsing_issue", url=response.url, issue_number=issue_number, title=issue_title)

        articles = response.css('table[class="tocArticle"]')
        log.info("articles_found", count=len(articles), url=response.url)

        for article in articles:
            article_link = article.css('div[class="tocTitle"] a::attr(href)').get()
            article_title = article.css('div[class="tocTitle"] a::text').get()
            article_number = article_link.split("/")[-1]

            yield response.follow(
                article_link,
                callback=self.parse_article,
                cb_kwargs={
                    "issue_number": issue_number,
                    "issue_title": issue_title,
                    "article_title": article_title,
                    "article_number": article_number,
                },
            )

    def parse_article(self, response, issue_number, issue_title, article_title, article_number):
        log.info("parsing_article", url=response.url, title=article_title)
        url = response.url
        pdf_url = response.css("a[class='file']::attr(href)").get()
        download_url = pdf_url.replace("view", "download") if pdf_url else None
        doi = response.css("a[id='pub-id::doi']::attr(href)").get()

        abstract_parts = response.css("div[id='articleAbstract'] div ::text").getall()
        abstract = " ".join([text.strip() for text in abstract_parts if text.strip()]) if abstract_parts else None

        keywords_raw = response.css("div[id='articleSubject'] div::text").get()
        keywords = [k.strip() for k in keywords_raw.split(";")] if keywords_raw else []

        authors_raw = response.css("div[id='authorString'] em::text").get()
        authors = [a.strip() for a in authors_raw.split(",")] if authors_raw else []

        yield {
            "issue_number": issue_number,
            "issue_title": issue_title,
            "article_title": article_title,
            "article_number": article_number,
            "url": url,
            "pdf_url": pdf_url,
            "download_url": pdf_url.replace("view", "download"),
            "doi": doi,
            "abstract": abstract,
            "keywords": keywords,
            "authors": authors,
        }
