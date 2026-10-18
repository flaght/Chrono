"""03跨期Live独立离线验收；运行前检查源码摘要，不连接MD/TD/数据库。"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "PF2026101209_REV6"

# 摘要内嵌，远端不需要额外.sha256文件；入口自身摘要只作运行审计。
EXPECTED_SHA256 = {
    "bomber/framework/datahub/__init__.py": "35c231a4208a091de6e95bb0397c337888165485da041e900e0dbb922292150d",
    "bomber/framework/datahub/basic.py": "c27f973f56b1a79f299d173b4810dc881c823d344a4cafbc873167b92f58634d",
    "bomber/framework/datahub/core.py": "df8b649a5e90fce1d076da11d3925c3260599ee1f880d4df73c7a9f17d1b466c",
    "bomber/framework/datahub/features/__init__.py": "dc6c00b797e2b71025dcf4cc7b0a995f631dbcf6bff27d5e969a033d187b095c",
    "bomber/framework/datahub/features/alignment.py": "18274df424302b4253fdbc250906d6c86de482fb55c4aa9021b923ad8610b099",
    "bomber/framework/datahub/features/contracts.py": "dd2155227778d6b19135a47daf445ce4621fe8749a3c237b3dbadcc3b74d1409",
    "bomber/framework/datahub/features/frames.py": "ad31385053862a057144e96e079eebb2f04db78cbd4cc9549f66743c3f04cdcc",
    "bomber/framework/datahub/features/replay.py": "0988404db8e9b2da2e6ee3936e4a0076e7e63262b8b50cbd945e61d7696a13b4",
    "bomber/framework/datahub/features/schema.py": "fa1ff693062957fcf7a1cfa5172710229bc185466f154661c993c8af059627df",
    "bomber/framework/datahub/features/store.py": "4df163553bd64490172e8c6071707183c5d5a81911b6e8a97832625da66345d4",
    "bomber/framework/datahub/future_basic.py": "895819a75f32eebd5c9f28c867c8bfa9cf487b287eaeb4b766f6a4078ecc44d0",
    "bomber/framework/datahub/option_basic.py": "215a400f502e7bd1ad43cf52cbea24ce44508ba3be3286df76898d58f6967b8c",
    "bomber/framework/datahub/role_prices.py": "b24529be15d68283cec444212fa925f5a035ade6dbfbd0e374e5d66951bf3a45",
    "bomber/framework/datahub/sector_roles.py": "bf918baf4c7eeb3617c4dd8ed28ada6032f67a6855f9c66f282fea0749245311",
    "bomber/framework/datahub/target_schedule.py": "eb6e112c5d013d7a2355688bd965d0b9b792542240c3b1dcc3437a17a4242037",
    "bomber/framework/datahub/temporal.py": "62463da79376ef793031597d2a3f62fcc1c0d469cb6a7f6dbd533c6d89519968",
    "bomber/framework/dataprep/__init__.py": "e190da45860082e0893bbc849f4ec705d5d36dc272e067a3d915bf2fca6eb075",
    "bomber/framework/dataprep/bars.py": "35679a3cca2e170f91a8916ffb85a9f37191ca930ad1b9afe762a2bcfb03b43b",
    "bomber/framework/dataprep/basic.py": "20251fc0ae02bb425d014315c079511623163d1989ebbc02f9f8750342293771",
    "bomber/framework/dataprep/calendar.py": "ad97c156acf7dd915e9727e30110efb1e5826e41f3512051c423b7462771e7ec",
    "bomber/framework/dataprep/catalog.py": "8b7939846bdd820a766546e01619d32a60037ba0d6b05673c37d7812f0ade948",
    "bomber/framework/dataprep/contracts.py": "e034dcca94f79ef8f7b71048e9a69ea5659c0d83e0e3907d09733527fdf679d7",
    "bomber/framework/dataprep/factors.py": "74c52c2d366ba72ada5ac4c1ec2e468868f25dda7f43bafbe54909b1355c142c",
    "bomber/framework/dataprep/features/__init__.py": "23fc8b3ea3719e09ff1316eb129b03e4212f33a119979efbc89d3e69a5573dee",
    "bomber/framework/dataprep/features/base.py": "f739063883e78f835996ad0058642cf10b71d8c3c8c8d77b0a1937056f683855",
    "bomber/framework/dataprep/features/codec.py": "af47f15be55f68a04aebc92d611637f754dd20ea859457e6689be99fda74902b",
    "bomber/framework/dataprep/features/database.py": "127e6da8bcf137e2042085327baef34e6dfd2eb0223e766a8233d48546b9a9c3",
    "bomber/framework/dataprep/features/definition.py": "fe71ab0ffe97cc92831ee74cad13c8013ec587427df882b6d207fa414cd1a720",
    "bomber/framework/dataprep/features/dolphin_query.py": "89d4be92072aec75b9eecd03f9c4b0c7879b87ee9cd82b1eb413eee7e9da1615",
    "bomber/framework/dataprep/features/factory.py": "32c383f4b3221116f11635c8734c8f17ffc2bc880ee9dd21ecd8772a8e1f8df6",
    "bomber/framework/dataprep/features/file.py": "188daa6879a6e3fce86aa930ac63376f975c96c15f151ee7201f4b2c6afaed9d",
    "bomber/framework/dataprep/features/query.py": "3a94d8bc0735e64e60da485a1ea3683438c78946c8f6f2ffcffa4f9ef256523a",
    "bomber/framework/dataprep/features/raw.py": "64669b104d16dae8b4e4f595f90d195e5f8cce3decaa61bed9e740283cbee1b7",
    "bomber/framework/dataprep/features/samples.py": "3d82f143b6034d23f70b2dccf6a2ccbc8cdf9658a2a500173c1c082171efae5d",
    "bomber/framework/dataprep/features/service.py": "c70092bcce0ee5189853075e207bc090fb240e31b31bf9dec1e760121b6a44a1",
    "bomber/framework/dataprep/futures.py": "a11ecd53185ffb82fea87692dce0bf093e386104189153ea33dfba0cb3c1eea1",
    "bomber/framework/dataprep/history/__init__.py": "e5494ff2a75a429a9258110573dfcd6c89726da9eba62097f7b65be39d2ee980",
    "bomber/framework/dataprep/history/base.py": "36fe7fb47fdbf81f696442f92a4a78aa63a878b5c46a476523e4a2b2f5b68324",
    "bomber/framework/dataprep/history/dolphindb.py": "78269f1a46642fcdd5c52284c62be3099e838106b7fe55b9d2a6d1a52644c714",
    "bomber/framework/dataprep/history/factors.py": "4e60276364ea94fce330ece6dbb8c3d68020a07ceb3a81d319740bd69c78bf67",
    "bomber/framework/dataprep/history/factory.py": "f557666bc2689ed4be8bb394ff2366508bef090ce225724741629b317bbc948e",
    "bomber/framework/dataprep/history/file.py": "47bc89b338c71a471c660aae316d950c0259dc6d98927169d5a67cdd45dcfcb1",
    "bomber/framework/dataprep/history/service.py": "41de2cc979196545630da8a98ecba8a4b5333adb4bf83794f3c5e59c4680e1b9",
    "bomber/framework/dataprep/history/windows.py": "df4924806fd54ff1897f46dc0d2e104320cb8180759665f8639eb00f957624e6",
    "bomber/framework/dataprep/live_cache.py": "f3661d73b9e1b4bc4b7d4b0c8121e8c3eb0ee994cb60af679ea97724510d477e",
    "bomber/framework/dataprep/live_cached_role.py": "5f5bf986ec27f330368082bf5a4bb62af3d5cfcf78d1aa61f5361521f0fafbd7",
    "bomber/framework/dataprep/live_options.py": "3222662581df4c0729e2eb4a2e077c9d83938ba35b424cb49ad27481943c4f0c",
    "bomber/framework/dataprep/live_portfolio.py": "87bfe1d033b4b3efa4338010fbf368a9b262d71987738744da6a82071cf1e48c",
    "bomber/framework/dataprep/live_references.py": "abd1fd378163c7222caa0bf7beae7f93d1d76fff6acd6aa591b821ec503100dd",
    "bomber/framework/dataprep/live_role.py": "719aeaa0dca8e2b2fc9060fa52b6fc34965d8041c9074be32c107162f91a305c",
    "bomber/framework/dataprep/metadata.py": "4d40a16024546e201152d608d8378eeb39d34a511529fbd49cf055f2da6f384c",
    "bomber/framework/dataprep/paths.py": "96c77d7521b261b87c4ab711bba2c4d2e552af5acff6ede81bf2f4b5fc598aef",
    "bomber/framework/dataprep/reference_freshness.py": "88035ce71f0add2b125cb478060bd2b7e8977aa7eecdfec5b3eb59f1346e5083",
    "bomber/framework/dataprep/references.py": "cfb44c953fb63b851d12a9d2d442547eb7c158bbd37c0488859f9d8b88276aa0",
    "bomber/framework/dataprep/scenarios/__init__.py": "215efd2ce895e6bf09b135a76095dc271d0cab9481f82d617a8b9084a4de2cd8",
    "bomber/framework/dataprep/scenarios/fixed_contracts.py": "b6df1a22408dc54a6c8d3c579fd012067b71b8275e04133988be1f93706f2911",
    "bomber/framework/dataprep/scenarios/option_chain.py": "bf1281fed341fbffbe0f57baba414d8cfb46c1ef2afbdf4a5a0796a6d4575f85",
    "bomber/framework/dataprep/scenarios/role_futures.py": "6551bba35de1ce79fb7bdf076d2f8fc33ae0e85481ca323add5932ab2a2d2cb6",
    "bomber/framework/dataprep/schedules.py": "12b0b5494a920d1bf159f88d9f734c4a1167a4b50a7e6daa62a6235f5c7e2b8d",
    "bomber/framework/dataprep/session.py": "5b4103fb0e5c5ee22fe0f08893a694effe5b43f43b12f7bf1e5d2bc060a4be35",
    "bomber/framework/dataprep/sources/__init__.py": "5faaf74efa3176840e26c41c5fc44af17429e2a8f7ce390bf4c7092304e7443e",
    "bomber/framework/dataprep/sources/artifacts.py": "6c75fae36d65ad40a7578a350b49635e0610e813ee30425f230f08ff2556cdc5",
    "bomber/framework/dataprep/sources/base.py": "b7a81690422fdc91c74ffdcfe171ae3b782321a64e685f5aa6c0312b454a9c41",
    "bomber/framework/dataprep/sources/dolphindb.py": "7eafbf43f08ac0184e0fd9c31c805e7c9b114ffa0e4f4e6c7ad99b5467b14e8a",
    "bomber/framework/dataprep/sources/factory.py": "56a734d797f04ce6ff5e5e49c093f05e85fec4d128174d7ad9fefe80069a6e62",
    "bomber/framework/dataprep/sources/file.py": "92a011fdde97c0cb962553106eeba5b188d676a4c7d6a8f18666652264f4214c",
    "bomber/framework/dataprep/sources/normalize.py": "449028e8d454b6169324755bac3b632f013a3676aedb84b9bcd1cddeed128204",
    "bomber/framework/dataprep/sources/policy.py": "33697be3613a893a2efda11ca83ed6e808ba73599024cb347c355cfee4c68016",
    "bomber/framework/market/basic/__init__.py": "595fd09639b5c4c9bebedea9c3583b269d5179c96bb97a4c117ed5d6f1f578b3",
    "bomber/framework/market/basic/base.py": "6b7a7a27b7c57bd6f9421b88344c14937e8172a856bca297dc5906255a3d53ea",
    "bomber/framework/market/basic/custom_bar.pyx": "0560fce01cd14f4085ed06034f6b0e66860c890eb733ba1f9e43135997fc7194",
    "bomber/framework/market/basic/fast_factory.pyx": "c73b3c74709c8d8fb0cf017eb6db42cbb7899a3aaf7ea97a3d41f45966cc9c28",
    "bomber/framework/market/basic/setup.py": "37844d9126cedfba617e66e7bec8498f25528dbfe7b70ea31af4b24a0f5d3772",
    "bomber/framework/market/stream/__init__.py": "ed08bb118527214a872ad34a11867a471bf90e3639bcf71a9ab981fcf9013dee",
    "bomber/framework/market/stream/aggregation.py": "2527d1c8f098f97f9100937b431315d37af03e892d27547dfee53db0e9a283e0",
    "bomber/framework/market/stream/base.py": "68aed09b402b7413e94e365b37e0b78e3f33e535d6cbf9d817352c1a5a9f98c5",
    "bomber/framework/market/stream/bn/__init__.py": "b8a158156ecbb3961fc95de55fce894efe4de50273a2f8d0af407e7d95ccac59",
    "bomber/framework/market/stream/bn/feed.py": "f0c64ce7d23dc38fc0cfddad781927b46894885365e32ed4da6da2876f81f9e2",
    "bomber/framework/market/stream/ctp/__init__.py": "3d9a0c364ee326ac7d4e07485d61cabcf4d20a9b4f3ce9157b6544ed104cb17f",
    "bomber/framework/market/stream/ctp/converter.py": "157648c40bed2f5467cb9de68ad4ec2c8f971d62ebf10c77bd7f7c052ea7ee7e",
    "bomber/framework/market/stream/ctp/feed.py": "065a4effbabbf281f13c9d64723a2ae6d073bec6d01147894c147e20b081a34b",
    "bomber/framework/market/stream/dolphin/__init__.py": "0fb14cd1d469180f1ec3a46940ddb6e36e326ebc0304ed2582a9efb7efe34a31",
    "bomber/framework/market/stream/dolphin/config.py": "fe307e68a768f156e381b5684d804450f353e4dfa771607beb03bd8784549c8e",
    "bomber/framework/market/stream/dolphin/converter.py": "f1f8393cc7707921a65d8ed7eb8dcd1442865a574c0e261679e27dc5306b49df",
    "bomber/framework/market/stream/dolphin/feed.py": "25e4cc7ea036ea2280a37d903808253f6948504e486029e49872fa5ab13da1b1",
    "bomber/framework/market/stream/feature_query.py": "46efd38358805bf460624f82bc1a89b8371c1107ec50dabd5d1ad1cca498179e",
    "bomber/framework/market/stream/health.py": "9db27606e1292e9e5a35c23ea2a32de466a0622effd960868e68d6a44353b8d1",
    "bomber/framework/trader/__init__.py": "9a05a30c35ae961f932f369747e6109a0d39d291a8700580313a2f4763bcbd3d",
    "bomber/framework/trader/assembly.py": "1dcc562546a61e3cddc68b962e9d9d79fc27278d58cb4be07c8570a692d5c0bc",
    "bomber/framework/trader/bar_sync.py": "fdbab53b432a9ae66c0dc5c3a1e0c31d5e14bab93ff472cffe3dd487e555ab91",
    "bomber/framework/trader/bridge/__init__.py": "383ba41bd13a70499069562d30879502ded5a53bf67e632b99a2b7165d67263d",
    "bomber/framework/trader/bridge/nautilus.py": "42d5e16324c7ce599fa153246478b969bd20999a651e5d78654ac7801bcfe352",
    "bomber/framework/trader/bridge/nautilus_events.py": "fbb489c131d9128775bf73ff6ba562921eae81082226c347d3d173df5a88d02f",
    "bomber/framework/trader/contracts.py": "becd8d8af550c5af3c5525ed0ffbc5491e008cd3d43725084038590585a1f609",
    "bomber/framework/trader/dynamic_routes.py": "b594985e66921baea088c30bf9d199e819cc4cce17e0b7564ea064c1311f37a5",
    "bomber/framework/trader/execution/__init__.py": "6d01f91fb58c9e40104d39c4d9e8d8d91d9314600a15219e7bb0b0f3887b30f0",
    "bomber/framework/trader/execution/account.py": "08e2bbafe0116c1a8672131c67b71cd7ea4f148c83db1d3dd63cfb242fd2fc72",
    "bomber/framework/trader/execution/attribution.py": "657b1ff48ce06f9f7e4e42116b27a7bf8237a4e4d8fd254cece5a0aa19ae0051",
    "bomber/framework/trader/execution/builders.py": "54733c8ab2e7bb9713a2cd2bfcd4d9fe274c53479bc7eac6a8661b9428d2b37d",
    "bomber/framework/trader/execution/contracts.py": "be18b7fc201ed16a2120b045b36df21211ce690b64b18a164b50d1414e6f512b",
    "bomber/framework/trader/execution/ctp/__init__.py": "b911cdf2f2afc4597f79725dd1269e1b3df362e47eacf8b56035dd7d3e4132c0",
    "bomber/framework/trader/execution/ctp/accounting.py": "6bb37ceb2bcad91b4c82f6a6cf77da0440cc8403867a3353f3417106c952f162",
    "bomber/framework/trader/execution/ctp/ledger.py": "97368cdee2e0411ff927e4bdb614e35b705980db5691cdbf488332bde135827e",
    "bomber/framework/trader/execution/ctp/limit_planner.py": "6e18f4a0de6cc83a1b34b33a91220f50421064c71a3ad2a4fa24b58a02452b1a",
    "bomber/framework/trader/execution/ctp/native_driver.py": "620f3847b53fac28ab85e25805ab961c58ac2e672b78f4b5eebe3f6c85f6f798",
    "bomber/framework/trader/execution/ctp/native_order.py": "a06a1d521934950dab379b2caf7e1ab2a251d0aecf5c238403f0098c0860e142",
    "bomber/framework/trader/execution/ctp/planner.py": "3d119db17d95b85289ba905c1d7f1fa5a3d97d7d54e31fadd5cb7d2189a15f40",
    "bomber/framework/trader/execution/ctp/profile.py": "879e053665fcc131da420b34ba71dc60f2e0efabaddccdee43ea870d549ca51b",
    "bomber/framework/trader/execution/ctp/td_transport.py": "75522647bc5b82226dbc1992e2958d4391f8920b724077062210c1dc42b71eb5",
    "bomber/framework/trader/execution/events.py": "e401ec27ae35afd2cf0d0e7d76b9edb450aa3b6899597d498ef8095e8741edcd",
    "bomber/framework/trader/execution/fanout.py": "4967e51eadb7a935b8b0758ddc5b9abf22efa3226dd6f3e0036bc6f08afff68f",
    "bomber/framework/trader/execution/live/__init__.py": "ae460805cbd46715816ea43d149edf590abd7c90cf536b27328dc2fe2c1193a7",
    "bomber/framework/trader/execution/live/account_reader.py": "9ce8ed496109a6399fec9b7411b62d3d871a8a8e31a37125a70a01c50dc9c7cb",
    "bomber/framework/trader/execution/live/backend.py": "4821cf02b5aa6b0c1e14a2425a4c2b4c652296e18dd35e3aa0e7c9a54acdfed8",
    "bomber/framework/trader/execution/live/binance_account.py": "b81ec72a8525c58dc00e1b843d7bd3365e49ad00a93e5db8450296a54d5399ba",
    "bomber/framework/trader/execution/live/binance_joint.py": "d277fd8dff89e9b72fae6fe8f6e4c9fe84f099d719682401aa40f70f864705b4",
    "bomber/framework/trader/execution/live/binance_orders.py": "2d5cf746f88b9ed963b246fa0bfa1582a4a028f89cddcae1e1a465f200af0836",
    "bomber/framework/trader/execution/live/client.py": "43af6872d10a7f2917646b0bb18719f36fd6d691c3dfb5e31fda96c2c9c25d8e",
    "bomber/framework/trader/execution/live/controlled.py": "aae7d666c0c39168663b3477ade136c7528324a2316a700706c03d253eab8942",
    "bomber/framework/trader/execution/live/driver.py": "a961fec4742fdafd2b0bdeda12b3f1c727f608bccb9f346e0a65be6877eb45e5",
    "bomber/framework/trader/execution/model.py": "84a0630300fa9444e026831a646c854449e7297b58c1d6212a281c41d562d545",
    "bomber/framework/trader/execution/nautilus.py": "1c9b67786df729419f1941457301c963f0faaf31f353807758068bda05f6320e",
    "bomber/framework/trader/execution/order_state.py": "88f9a47bbe3b202f3c25d8b1a885e9d6138068c1efb8f444aabecb933e467ce4",
    "bomber/framework/trader/execution/planner.py": "86cfd7a3cbfffe999cbc30b3c0c2374ef2788380ee0310e6b17ab40a15116e2f",
    "bomber/framework/trader/execution/policy.py": "80f8e793d3a5362ee9d10f17193d5788607cdbad044d89cfa9a21624b742a1fe",
    "bomber/framework/trader/execution/ports.py": "b4b3b92b65e2c2215570a4bd09c42b7d7c6669d26acf7b87a41ebfcb3f758aed",
    "bomber/framework/trader/execution/recording.py": "5fda42ecf66f85cded2f162a5b674d671539b30fcd95f86331de83b5504bfc96",
    "bomber/framework/trader/execution/risk.py": "757047f03f8e1db22a7c4a8b053c72b08011dee631842bedf478790ef5c1e48d",
    "bomber/framework/trader/execution/simulation/__init__.py": "82c69aad7d509c19f2d67c265d14d6a2cfa781eacfc5c5d87a3226898fcd18ae",
    "bomber/framework/trader/execution/simulation/backend.py": "291eac4d450bb6373f527bb045fa86ebc22fc54d732f16494a1599e15f59f22a",
    "bomber/framework/trader/execution/simulation/binance.py": "853a96e9f7bc1656b342d3e5ab95f7be64e638a209583dd8a0e2083aa48217b4",
    "bomber/framework/trader/execution/simulation/client.py": "ce61b0b6222b48a3e4b89bb760d02fc46676df1bb3569f5172bb230a0919d7df",
    "bomber/framework/trader/execution/simulation/ctp.py": "2103c2324637463ead3e748f2256e5be0f00174c5a60c577cbfbf8f3ba5a32ac",
    "bomber/framework/trader/execution/simulation/gateway.py": "75a79ea7b29c07f8e1a5c03cd95fcc2a753044dcd487772e0f9e7c7bd41b338d",
    "bomber/framework/trader/execution/simulation/profile.py": "ae3211481ca65740da8905c9d747544899db4ec4a4173ef114717e5126730a01",
    "bomber/framework/trader/instrument_factory.py": "ad16d44329dbf504f96d4ded401d19035f7e9f00ec234ed4a098f6423bf0808e",
    "bomber/framework/trader/live_roles.py": "d734725824d28773b43287b1f3e327eeaa4e3e3537c64bcb16269becc4fbe477",
    "bomber/framework/trader/market_health.py": "ebfdbf57cd84317c4993d11a76a5b5027a7c12a1102bd518e25b01ff4e246850",
    "bomber/framework/trader/persistence.py": "72b593923da819b71927fc0ed5d598fc66ad28c1b2eb6991e0561f0554668ea6",
    "bomber/framework/trader/portfolio.py": "1961c4319f29285a8d2782492f7c951bd890a6b0f786a234c5cd9664cb499d61",
    "bomber/framework/trader/ports.py": "188f4dc83c1605f865eb0a870bbbfa58839e672f427843b8c0c510bb23812c09",
    "bomber/framework/trader/runner.py": "652abad3f186ad18d30a02a6ea078846c45be405b8f038a309ff947e93a2493d",
    "bomber/framework/trader/runtime/__init__.py": "ab4bb70aac91ff57cfbc3ad469401e9d1848635379f6dc21bf2bed1e2f14c258",
    "bomber/framework/trader/runtime/base.py": "5e45e081e469decc196a2f0f8c09763108dec1f2e6100671268b29aa6b7ac1aa",
    "bomber/framework/trader/runtime/ctp.py": "a0b1f5e1cabfb781babbf7fcbc7bf38b6104f9889d095ad72c0c16014d985afc",
    "bomber/framework/trader/runtime/ctp_account.py": "a404376dd009fd1f777a61f2683bf35723bf18b884da635aa316e94c2724fc6a",
    "bomber/framework/trader/runtime/ctp_positions.py": "1bb5d64fbf73cfb9fade5a216841fd52ebfa773ac262f9eebf88dfafc5407c7d",
    "bomber/framework/trader/runtime/ctp_setup.py": "dbe26b11053592f64c6d42c41a633f6e44be6f1e90bfb4a31d612ae2fd8fb20f",
    "bomber/framework/trader/runtime/direct.py": "a8e648849f547609c60f61ee7bbd68866be54fe13b44a16aee3dff1d4dd3408b",
    "bomber/framework/trader/runtime/historical.py": "c0f53aa8cec878ad0439cc3197e5ce70c3116a8af69c09b57c6d0623f2409c44",
    "bomber/framework/trader/runtime/joint.py": "a5b17f9339d0d2a6895b7297c26aeb2746be7e8ad3106c1d2394d73b8b4ace93",
    "bomber/framework/trader/runtime/live/__init__.py": "65945a33758bd8148efd183312e4563d7f915d11971b4c35b34e47c2980c2cfd",
    "bomber/framework/trader/runtime/live/channels/__init__.py": "73941f4c92159d277d4f3e719b63eb2ffbeee7ff92197384e167fe34861023dd",
    "bomber/framework/trader/runtime/live/channels/binance.py": "5ec6f41640bb7f4980572a5327f313df9fa05b5f6878822a0090b823980ba820",
    "bomber/framework/trader/runtime/live/channels/ctp.py": "cdc26eed4531d568e716b5aa6141b4f5d723e09dae32d06654f02608f1e927bf",
    "bomber/framework/trader/runtime/live/channels/ctp_market.py": "2b7ad1b8ad805fbb5edd30f16897da591e5ae1ec0d22505c84be168517d60f01",
    "bomber/framework/trader/runtime/live/channels/ctp_recovery.py": "8764545ae3ba22ddb5cbd242ca3447fa5aa569a9615bbde90737c883584339e8",
    "bomber/framework/trader/runtime/live/channels/dolphin.py": "bb725c4b780ba8d65a20298e267ec33161251c15f0bdc121e83bfcb6082895a3",
    "bomber/framework/trader/runtime/live/channels/features.py": "bc3fba69836682cd6cf5d71014c2f99cb6182f6011bb80fcfa5125f667c55288",
    "bomber/framework/trader/runtime/live/contracts.py": "f44f2c2b3d4ae0a698b333fa50a34eda5f3b786977eea8912156f340c38d90ec",
    "bomber/framework/trader/runtime/live/health.py": "d2cbdd334c58511b2acfe03eb8f8ef2858dbaec7bebcb44bed9c61a29a608129",
    "bomber/framework/trader/runtime/live/local.py": "3f803ed93001127250336b3c76ab12267e7537a01b7f8f3b0480206e638a2a53",
    "bomber/framework/trader/runtime/live/profiles.py": "3cab9e019376d7e7c1db7976e0a1510198bb49c947e23d18adf658cbb783900c",
    "bomber/framework/trader/runtime/live/readiness.py": "125a27f8b9ad3ede9809fa0865dfac90f44572f7e979f22c1e115db19883e96b",
    "bomber/framework/trader/runtime/live/recovery.py": "a566673369712a330d27c3dcb3effe06211c5c049a162ccf97a5fb753c197758",
    "bomber/framework/trader/runtime/live/runtime.py": "9d582ceaa00f9f7c2105b908abc501afe1abfae3d1b9b997188a6108792f1fc1",
    "bomber/framework/trader/runtime/live_policy.py": "06d29a9a5a0827987a720bcf0ab2a46fec8506a6d5f6a7182c22f62830f65647",
    "bomber/framework/trader/runtime/managed.py": "2d873855d0645f90bfdac5ea32d94a2e1d25cb38e135c2d7c5fd0cce3ce37857",
    "bomber/framework/trader/runtime/market_adapter.py": "0d49ce7e15e6b0438b5ad6b0f8f6b07fc20ea7eb8a9b332978d073af111a0b21",
    "bomber/framework/trader/runtime/nautilus.py": "00a2988872118f00a1b5e0688f6633ab92328eb18d1467ebc152b40b97113a08",
    "bomber/framework/trader/runtime/ownership.py": "33899be4cd5cf287ca3ec1e04a8cbad6af33776dec1b43195a04fcdfc4dcdd96",
    "bomber/framework/trader/runtime/replay.py": "8083ef2459abf364029af3db2f595591755dd245f9050802d815665874528d5c",
    "bomber/framework/trader/runtime/reports.py": "7dc4c9f526c165a99f9d52a777dd9cd89628c950d3c8698d7e66052ccb674b3e",
    "bomber/framework/trader/runtime/trading_sessions.py": "5defb3c36ed4dca7aead64e426d5db33126aa9b875aa60959539d36686c65efd",
    "bomber/framework/trader/scheduling/__init__.py": "4cee3f98b63418abefb4383676e86032d2ee7e22a25fde9e04f2f8b551b2c6ea",
    "bomber/framework/trader/scheduling/clock.py": "4ea0f0293aed9c45fa7354ee56d8168e48b380dad80f694c3710cb03f2feaeed",
    "bomber/framework/trader/template.py": "1bdbecbc9b9ca83bb35d62500b8ecde653120c2875ea0ace44e4e47ca3463e29",
    "demos/03_calendar_spread/calendar_signal.py": "4fa04d81b96232cd87435f005ddd453dcf98f0213da064f8e3c31a4ced0e5c5b",
    "demos/03_calendar_spread/live_references.py": "37947577572f4d762aaf00da27a7856abc1ef20ec2fe66f40299a181466fee2b",
    "demos/03_calendar_spread/live_runtime.py": "b816dd230e05ac9ae6be30dd51496590a4368e0fce6b75091713e5a979d6fec1",
    "demos/03_calendar_spread/recovery.py": "4cb36c5a21e4df7b3c9296c9b40d67b840397fc3b3f891d74d7dbec3c4791721",
    "demos/03_calendar_spread/run_live.py": "023d7a51e5a258c5d3ef280c1d4952a32b872d4a7bc2654dd9f116b5de5b30c3",
    "demos/03_calendar_spread/strategy.py": "ddf4f6d3ec4c90397c9305a5aaa36faa23604d28a9f6f180e4ad53fb7effa98e",
    "tests/run_p4_ctp_td_transport.py": "41227fcf46462e18a1267afd1e451177f72dafa4dec03877daa25c81ce2f6fb1",
    "tests/test_calendar_spread_live.py": "4d5dbf746d849b525de9d7ff646b77a138a42c3c472fc4597b70e0d3b3ce6cab",
    "tests/test_calendar_spread_selection.py": "2fec339f43096fe3a1303607a96567c182c053b4b6d99767e3c90d84a950ae63",
    "tests/test_execution_composition.py": "c6fd68580b0b31325f5825310622066f37e87017d0366e6af16b06362ccbd171",
    "tests/test_live_channels.py": "4f1b36c3bbe3b5c4948a7e8912398f7840a4d5cd39768b20745f7890a1db5d8d",
    "tests/test_live_common.py": "718871a1902621f8da124abd8019f70188599a6ee83ef59b5c2041adbb5cd5e5",
    "tests/test_live_recovery.py": "3d536ee48fbcc9fda1a91de95d8f65b2350e1fec00ed8eebe5295db8062101c6",
    "tests/test_calendar_spread_acceptance.py": "2c0e09fdc8558b6eb9c50058a9f5fe003ad258dc655be5e1ad4247ad30d8a833",
}
GROUPS = {
    "entry": ("unittest", "-v", "tests.test_calendar_spread_acceptance"),
    "live": ("unittest", "-v", "tests.test_calendar_spread_live"),
    "selection": ("tests.test_calendar_spread_selection",),
    "public": ("unittest", "tests.test_live_common", "tests.test_live_channels",
               "tests.test_live_recovery", "tests.test_execution_composition"),
}


def check_sources():
    failures = []
    for relative, expected in EXPECTED_SHA256.items():
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
        if actual != expected:
            failures.append({"file": relative, "expected": expected, "actual": actual})
    if failures:
        raise RuntimeError("跨期Live源码缺失或版本不符:\n" + json.dumps(failures, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", choices=tuple(GROUPS))
    parser.add_argument("--report-dir", type=Path, default=ROOT / "tests" / "results")
    args = parser.parse_args(argv)
    # 此前置只用标准库，原生模块及测试在摘要检查完成后由子进程导入。
    check_sources()
    selected = tuple(dict.fromkeys(args.only or ("entry", "live", "selection")))
    destination = args.report_dir / f"calendar-spread-acceptance-{time.time_ns()}"
    destination.mkdir(parents=True, exist_ok=False)
    entry_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(f"Calendar Live baseline={BASELINE} root={ROOT} python={sys.executable} entry_sha256={entry_sha}", flush=True)
    rows = []
    for name in selected:
        command = [sys.executable, "-u", "-m", *GROUPS[name]]
        start = time.monotonic()
        process = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (destination / f"{name}.log").write_text(process.stdout, encoding="utf-8")
        print(process.stdout, end="", flush=True)
        rows.append({"group": name, "exit_code": process.returncode, "seconds": time.monotonic() - start})
    result = {"baseline": BASELINE, "root": str(ROOT), "python": sys.executable, "entry_sha256": entry_sha,
        "status": "passed" if all(row["exit_code"] == 0 for row in rows) else "failed", "groups": rows,
        "network_connections": False, "counter_fills_evidence": False}
    (destination / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{result['status']}: {destination}", flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
