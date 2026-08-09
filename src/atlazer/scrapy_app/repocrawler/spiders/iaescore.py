import structlog
import scrapy

from typing import Dict, Any

log = structlog.get_logger(__name__)


class IaeScoreSpider(scrapy.Spider):
    name = "iaescore"

    def __init__(
        self,
        journal: str,
        issue_number: str = "",
        article_number: str = "",
        issue_done: bool = False,
        article_done: bool = False,
        page: str = "1",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.journal = journal
        self.issue_number = int(issue_number) if issue_number else None
        self.article_number = int(article_number) if article_number else None
        self.issue_done = issue_done
        self.article_done = article_done
        self.page = int(page)
        self.start_urls = [f"https://{journal}.iaescore.com/index.php/{journal.capitalize()}/issue/archive?issuesPage={self.page}"]

        log.info(
            "spider_initialized",
            journal=self.journal,
            issue_number=self.issue_number,
            article_number=self.article_number,
            page=self.page,
            start_urls=self.start_urls,
        )

    def start_requests(self):
        """Di sini page bisa di-override sebelum request pertama dieksekusi,
        misal berdasarkan state terakhir di redis/postgres."""

        url = (
            f"https://{self.journal}.iaescore.com/index.php/"
            f"{self.journal.capitalize()}/issue/archive?issuesPage={self.page}"
        )

        log.info("start_request_built", url=url, page=self.page)
        yield scrapy.Request(url, callback=self.parse)

    def closed(self, reason):
        log.info("spider_closed", reason=reason)

    def parse(self, response):
        """Disini cek apakah issue sudah selesai di proses atau belum.
        Tidak mengunjungi semua link paper didalam issue. Satu per-satu."""

        issues = response.css('div[id^="issue-"]')
        processing_issues = []
        log.info("issues_found", count=len(issues), url=response.url)

        for issue in issues:
            issue_id = issue.attrib.get("id")           # "issue-588"
            issue_number = int(issue_id.split("-")[1])  # 588
            link = issue.css("h4 a::attr(href)").get()
            title = issue.css("h4 a::text").get()

            processing_issues.append({
                "number": issue_number,
                "title": title,
                "link": link,
            })

        log.info("issues_found", issues=issues)

        # filter out from raw_issues if the issue is done
        if self.issue_done:
            processing_issues = [s for s in processing_issues if int(s["number"]) != self.issue_number]

        log.info("processing_issues", issues=processing_issues)

        # langsung proses ke halaman selanjutnya
        # if self.issue_done:
        #     self.page = self.page + 1
        #     log.info("issue_done", issue_number=self.issue_number, next_page=self.page)
        #     url = f"https://{self.journal}.iaescore.com/index.php/{self.journal.capitalize()}/issue/archive?issuesPage={self.page}"
        #     yield response.follow(url, callback=self.parse)
        #     return

        # get first row only
        processing_issues = processing_issues[:1]
        if len(processing_issues) > 0:
            self.issue_number = processing_issues[0]["number"]

            for issue in processing_issues:
                log.info("issue_matched", issue_number=issue["number"], title=issue["title"], url=issue["link"])
                yield response.follow(
                    issue["link"],
                    callback=self.parse_issue,
                    cb_kwargs={
                        "issue_number": issue["number"],
                        "issue_title": issue["title"],
                        "processing_issues": processing_issues,
                    },
                )
                return

        # lanjut ke halaman pagination berikutnya
        # next_pages = response.css('a[href*="issuesPage"]::attr(href)').getall()
        # if next_pages:
        #     # ambil link "next" terakhir yang unik (hindari duplikasi karena ada link '>' dan '>>')
        #     for np in next_pages:
        #         yield response.follow(np, callback=self.parse)

    def parse_issue(self, response, issue_number, issue_title):
        log.info(
            "parsing_issue",
            url=response.url,
            issue_number=issue_number,
            title=issue_title,
        )

        articles = response.css('table[class="tocArticle"]')
        log.info("articles_found", count=len(articles), url=response.url)

        processing_articles = []
        issue_done: bool = False

        for article in articles:
            link = article.css('div[class="tocTitle"] a::attr(href)').get()
            title = article.css('div[class="tocTitle"] a::text').get()
            number = link.split("/")[-1]

            processing_articles.append({
                "number": int(number),
                "title": title,
                "link": link,
            })

        log.info("processing_articles", articles=processing_articles)

        # filter out from raw_articles if the article is done
        # hilangkan semua articles setelah `article_number` yang terakhir
        # ex = [1, 2, 3]
        # article_number = 2
        # result = [3]
        if self.article_number:
            processing_articles = self._get_remaining_after(
                processing_articles=processing_articles,
                article_number=self.article_number,
            )

        log.info("processing_articles", articles=processing_articles)

        # apabila tidak ada artikel lagi maka current issue telah selesai
        if len(processing_articles) <= 0:
            issue_done = True

        # get first row only
        processing_articles = processing_articles[:1]
        if len(processing_articles) > 0:
            self.article_number = processing_articles[0]["number"]

            for article in processing_articles:
                yield response.follow(
                    article["link"],
                    callback=self.parse_article,
                    cb_kwargs={
                        "issue_number": issue_number,
                        "issue_title": issue_title,
                        "article_title": article["title"],
                        "article_number": article["number"],
                        "issue_done": issue_done,
                    },
                )

    def parse_article(
        self,
        response,
        issue_number,
        issue_title,
        article_title,
        article_number,
        issue_done,
    ):
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
            "paper": {
                "issue_number": issue_number,
                "issue_title": issue_title,
                "article_title": article_title,
                "article_number": article_number,
                "url": url,
                "pdf_url": pdf_url,
                "download_url": download_url,
                "doi": doi,
                "abstract": abstract,
                "keywords": keywords,
                "authors": authors,
            },
            "next_page": self.page,
            "next_issue_number": self.issue_number,
            "next_article_number": self.article_number,
            "issue_done": issue_done,
        }

    def _get_remaining_after(self, processing_articles: list, article_number: int) -> list:
        """
        Mengembalikan elemen-elemen setelah dictionary dengan "number" == article_number.

        Contoh:
            processing_articles = [{"number": 1, "title": "abc"}, {"number": 2, "title": "axy"}, {"number": 3, "title": "def"}]
            article_number = 2
            result = [{"number": 3, "title": "def"}]
        """
        idx = next(
            (i for i, item in enumerate(processing_articles) if item.get("number") == article_number),
            None,
        )

        if idx is None:
            # article_number tidak ditemukan
            return []

        return processing_articles[idx + 1:]
